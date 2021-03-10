/* Edge Impulse inferencing library
 * Copyright (c) 2021 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "../ei_classifier_porting.h"
#if EI_PORTING_ECM3532 == 1

#include <stdarg.h>
#include <stdlib.h>
#include "eta_bsp.h"
#include "FreeRTOS.h"

__attribute__((weak)) EI_IMPULSE_ERROR ei_run_impulse_check_canceled() {
    return EI_IMPULSE_OK;
}

/**
 * Cancelable sleep, can be triggered with signal from other thread
 */
__attribute__((weak)) EI_IMPULSE_ERROR ei_sleep(int32_t time_ms) {
    EtaCspTimerDelayMs(time_ms);
    return EI_IMPULSE_OK;
}

uint64_t ei_read_timer_ms() {
    return EtaCspTimerCountGetMs();
}

uint64_t ei_read_timer_us() {
    return 0;
}

__attribute__((weak)) void ei_printf(const char *format, ...) {
    
    extern tUart etaUart;
    char print_buf[1024] = {0};

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        EtaCspUartPuts(&etaUart, print_buf);
    }
}

__attribute__((weak)) void ei_printf_float(float f) {
    ei_printf("%f", f);
}

__attribute__((weak)) void *ei_malloc(size_t size) {
    return pvPortMalloc(size);
}

__attribute__((weak)) void *ei_calloc(size_t nitems, size_t size) {
    
    uint32_t ix;
    uint8_t *ptr = (uint8_t *)pvPortMalloc(nitems * size);

    if(ptr) {
        for (ix = 0; ix < (nitems * size); ix++) {
            *(ptr + ix) = 0;            
        }
    }

    return (void *)ptr;
}

__attribute__((weak)) void ei_free(void *ptr) {
    vPortFree(ptr);
}

#if defined(__cplusplus) && EI_C_LINKAGE == 1
extern "C"
#endif
__attribute__((weak)) void DebugLog(const char* s) {
    ei_printf("%s", s);
}

#endif // EI_PORTING_ECM3532 == 1
