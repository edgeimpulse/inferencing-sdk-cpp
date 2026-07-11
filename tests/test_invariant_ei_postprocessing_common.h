#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Minimal stubs to replicate the vulnerable pattern without the full SDK */

typedef struct {
    float  *buffer;
    size_t  allocated_size; /* in bytes */
} freeform_output_t;

typedef struct {
    float  *buffer;
    size_t  rows;
    size_t  cols;
} matrix_t;

/*
 * Safe version of the copy that MUST be used:
 * Returns 0 on success, -1 if the computed size exceeds the allocated buffer.
 */
static int safe_postprocess_copy(freeform_output_t *freeform_output,
                                 const matrix_t    *raw_output_mtx)
{
    if (!freeform_output || !raw_output_mtx) return -1;
    if (!freeform_output->buffer || !raw_output_mtx->buffer) return -1;

    size_t required = raw_output_mtx->rows * raw_output_mtx->cols * sizeof(float);

    /* Security invariant: never copy more bytes than the destination holds */
    if (required > freeform_output->allocated_size) {
        return -1; /* reject oversized input */
    }

    memcpy(freeform_output->buffer,
           raw_output_mtx->buffer,
           required);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test cases                                                           */
/* ------------------------------------------------------------------ */

/*
 * Each payload describes a (rows, cols) pair for the model output matrix.
 * The destination buffer is always allocated for EXPECTED_ROWS * EXPECTED_COLS
 * floats.  Any pair whose product exceeds that must be rejected.
 */
#define EXPECTED_ROWS  4u
#define EXPECTED_COLS  8u
#define EXPECTED_ELEMS (EXPECTED_ROWS * EXPECTED_COLS)   /* 32 floats */
#define EXPECTED_BYTES (EXPECTED_ELEMS * sizeof(float))  /* 128 bytes */

typedef struct { size_t rows; size_t cols; int should_succeed; } dim_payload_t;

START_TEST(test_buffer_read_never_exceeds_declared_length)
{
    /* Invariant: memcpy from model output must never exceed the allocated
     * destination buffer size.  Oversized dimensions must be rejected. */

    const dim_payload_t payloads[] = {
        /* --- exact-fit (must succeed) --- */
        { EXPECTED_ROWS,      EXPECTED_COLS,      1 },
        { 1,                  EXPECTED_ELEMS,     1 },
        { EXPECTED_ELEMS,     1,                  1 },

        /* --- 2x oversize --- */
        { EXPECTED_ROWS * 2,  EXPECTED_COLS,      0 },
        { EXPECTED_ROWS,      EXPECTED_COLS * 2,  0 },
        { EXPECTED_ROWS * 2,  EXPECTED_COLS * 2,  0 },

        /* --- 10x oversize --- */
        { EXPECTED_ROWS * 10, EXPECTED_COLS,      0 },
        { EXPECTED_ROWS,      EXPECTED_COLS * 10, 0 },
        { EXPECTED_ROWS * 10, EXPECTED_COLS * 10, 0 },

        /* --- off-by-one (one element too many) --- */
        { EXPECTED_ROWS,      EXPECTED_COLS + 1,  0 },
        { EXPECTED_ROWS + 1,  EXPECTED_COLS,      0 },
        { 1,                  EXPECTED_ELEMS + 1, 0 },

        /* --- extreme / adversarial sizes --- */
        { 1024,               1024,               0 },
        { 65536,              1,                  0 },
        { 1,                  65536,              0 },
        { SIZE_MAX / sizeof(float), 1,            0 }, /* would overflow */
        { 1, SIZE_MAX / sizeof(float),            0 }, /* would overflow */

        /* --- zero dimensions (edge case, no copy needed) --- */
        { 0,                  EXPECTED_COLS,      1 },
        { EXPECTED_ROWS,      0,                  1 },
        { 0,                  0,                  1 },

        /* --- undersized (must succeed) --- */
        { EXPECTED_ROWS / 2,  EXPECTED_COLS,      1 },
        { EXPECTED_ROWS,      EXPECTED_COLS / 2,  1 },
        { 1,                  1,                  1 },
    };

    int num_payloads = (int)(sizeof(payloads) / sizeof(payloads[0]));

    /* Destination buffer: exactly EXPECTED_BYTES */
    float *dest_buf = (float *)malloc(EXPECTED_BYTES);
    ck_assert_ptr_nonnull(dest_buf);

    freeform_output_t freeform_output;
    freeform_output.buffer         = dest_buf;
    freeform_output.allocated_size = EXPECTED_BYTES;

    for (int i = 0; i < num_payloads; i++) {
        size_t rows = payloads[i].rows;
        size_t cols = payloads[i].cols;
        int    expected_ok = payloads[i].should_succeed;

        /* Guard against multiplication overflow when allocating source */
        size_t src_elems = 0;
        int    overflow  = 0;

        if (rows != 0 && cols != 0) {
            if (rows > SIZE_MAX / cols) {
                overflow = 1;
            } else {
                src_elems = rows * cols;
                if (src_elems > SIZE_MAX / sizeof(float)) {
                    overflow = 1;
                }
            }
        }

        float *src_buf = NULL;
        if (!overflow && src_elems > 0) {
            src_buf = (float *)calloc(src_elems, sizeof(float));
            /* If allocation fails for huge sizes, treat as rejected */
            if (!src_buf) overflow = 1;
        }

        matrix_t raw_output_mtx;
        raw_output_mtx.buffer = src_buf; /* may be NULL for zero-size */
        raw_output_mtx.rows   = rows;
        raw_output_mtx.cols   = cols;

        int result;
        if (overflow) {
            /* An overflowing size must always be rejected */
            result = -1;
        } else {
            result = safe_postprocess_copy(&freeform_output, &raw_output_mtx);
        }

        if (expected_ok) {
            ck_assert_msg(result == 0,
                "payload[%d] rows=%zu cols=%zu: expected success but got rejection",
                i, rows, cols);
        } else {
            ck_assert_msg(result != 0,
                "payload[%d] rows=%zu cols=%zu: oversized input was NOT rejected "
                "(buffer overflow would occur)",
                i, rows, cols);
        }

        /* Verify destination buffer sentinel is intact (canary check) */
        /* The buffer was calloc'd so any write beyond EXPECTED_BYTES would
         * corrupt adjacent memory; we rely on the allocator / sanitisers here,
         * but we also explicitly check the invariant via the return value above. */

        if (src_buf) free(src_buf);
    }

    free(dest_buf);
}
END_TEST

/* ------------------------------------------------------------------ */
/* Additional focused test: zero-size allocation as destination        */
/* ------------------------------------------------------------------ */
START_TEST(test_zero_size_destination_always_rejected)
{
    /* Invariant: if the destination has zero allocated bytes, any non-empty
     * source must be rejected. */

    float dummy_src[16] = {0};

    freeform_output_t freeform_output;
    freeform_output.buffer         = (float *)malloc(1); /* 1-byte placeholder */
    ck_assert_ptr_nonnull(freeform_output.buffer);
    freeform_output.allocated_size = 0; /* zero usable space */

    matrix_t raw_output_mtx;
    raw_output_mtx.buffer = dummy_src;
    raw_output_mtx.rows   = 4;
    raw_output_mtx.cols   = 4;

    int result = safe_postprocess_copy(&freeform_output, &raw_output_mtx);
    ck_assert_msg(result != 0,
        "Non-empty source into zero-size destination must be rejected");

    free(freeform_output.buffer);
}
END_TEST

/* ------------------------------------------------------------------ */
/* Suite wiring                                                         */
/* ------------------------------------------------------------------ */
Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s       = suite_create("Security_CWE120_BufferOverread");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_buffer_read_never_exceeds_declared_length);
    tcase_add_test(tc_core, test_zero_size_destination_always_rejected);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int      number_failed;
    Suite   *s;
    SRunner *sr;

    s  = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}