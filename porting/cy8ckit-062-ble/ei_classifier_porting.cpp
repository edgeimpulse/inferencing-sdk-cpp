#include "../ei_classifier_porting.h"
#if EI_PORTING_CY8CKIT_062_BLE == 1

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include "timer.h"

#define EI_WEAK_FN __attribute__((weak))

EI_WEAK_FN EI_IMPULSE_ERROR ei_run_impulse_check_canceled() {
    return EI_IMPULSE_OK;
}

EI_WEAK_FN EI_IMPULSE_ERROR ei_sleep(int32_t time_ms) {
    Cy_SysLib_Delay(time_ms);
    return EI_IMPULSE_OK;
}

uint64_t ei_read_timer_ms() {
    if (!timer_initialized) {
        timer_init();
    }
    uint32_t cnt = cyhal_lptimer_read(&lptimerObj);
    return (uint64_t)cnt * 1000UL / CY_SYSCLK_WCO_FREQ;
}

uint64_t ei_read_timer_us() {
    return ei_read_timer_ms() * 1000UL;
}

/**
 *  Printf function uses vsnprintf and output using Arduino Serial
 */
__attribute__((weak)) void ei_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

__attribute__((weak)) void ei_printf_float(float f) {
    ei_printf("%f", f);
}

__attribute__((weak)) void *ei_malloc(size_t size) {
    return malloc(size);
}

__attribute__((weak)) void *ei_calloc(size_t nitems, size_t size) {
    return calloc(nitems, size);
}

__attribute__((weak)) void ei_free(void *ptr) {
    free(ptr);
}

#if defined(__cplusplus) && EI_C_LINKAGE == 1
extern "C"
#endif
__attribute__((weak)) void DebugLog(const char* s) {
    ei_printf("%s", s);
}

#endif // EI_PORTING_CY8CKIT_062_BLE == 1
