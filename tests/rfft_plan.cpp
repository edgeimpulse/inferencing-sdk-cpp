#include "edge-impulse-sdk/dsp/speechpy/feature.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

static size_t live = 0;
static bool fail_alloc = false;
void *ei_malloc(size_t n) { if (fail_alloc) return nullptr; void *p = malloc(n); if (p) ++live; return p; }
void *ei_calloc(size_t n, size_t s) { if (fail_alloc) return nullptr; void *p = calloc(n,s); if (p) ++live; return p; }
void ei_free(void *p) { if (p) --live; free(p); }
void ei_printf(const char *, ...) {}

int main() {
    {
        ei::numpy::rfft_plan plan;
        for (size_t n : {256, 256, 32, 20, 64, 256}) {
            for (int sample = 0; sample < 8; ++sample) {
                std::vector<float> input(n), a(n/2+1), b(n/2+1);
                for (size_t i=0; i<n; ++i) input[i] = sample ? sinf(i * .17f * sample) : 0;
                assert(ei::numpy::power_spectrum(input.data(),n,a.data(),a.size(),n) == 0);
                assert(ei::numpy::power_spectrum(input.data(),n,b.data(),b.size(),n,&plan) == 0);
                assert(memcmp(a.data(),b.data(),a.size()*sizeof(float)) == 0);
                auto c=input, d=input;
                assert(ei::numpy::dct2(c.data(),n,ei::DCT_NORMALIZATION_ORTHO) == 0);
                assert(ei::numpy::dct2(d.data(),n,ei::DCT_NORMALIZATION_ORTHO,&plan) == 0);
                assert(memcmp(c.data(),d.data(),n*sizeof(float)) == 0);
            }
        }
    }
    assert(live == 0);
    {
        ei::numpy::rfft_plan plan;
        float in[64] = {};
        ei::fft_complex_t out[33];
        fail_alloc = true;
        assert(ei::numpy::software_rfft(in,out,64,33,&plan) != 0);
        fail_alloc = false;
        assert(ei::numpy::software_rfft(in,out,64,33,&plan) == 0);
    }
    assert(live == 0);
    {
        ei::matrix_t matrix(4,20);
        float expected[80];
        for (size_t i=0; i<80; ++i) matrix.buffer[i] = expected[i] = cosf(i*.13f);
        for (size_t row=0; row<4; ++row)
            assert(ei::numpy::dct2(expected+row*20,20,ei::DCT_NORMALIZATION_ORTHO) == 0);
        assert(ei::numpy::dct2(&matrix,ei::DCT_NORMALIZATION_ORTHO) == 0);
        assert(memcmp(expected,matrix.buffer,sizeof(expected)) == 0);
    }
    assert(live == 0);
    puts("PASS: bit-identical spectra and DCT; size changes; allocation failure/retry; no leaks");
}
