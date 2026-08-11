#include "../ei_classifier_porting.h"
#if EI_PORTING_HIMAX_WE2 == 1
#include "WE2_core.h"

// Impelements weak functions from edge-impulse-sdk/porting/ethos-core-driver/src/ethosu_driver.c
void ethosu_flush_dcache(const uint64_t *base_addr, const size_t *base_addr_size, int num_base_addr)
{
    for (int ix = 0; ix < num_base_addr; ix++) {
        hx_CleanDCache_by_Addr((volatile void *)(uintptr_t)base_addr[ix], base_addr_size[ix]);
    }
}

void ethosu_invalidate_dcache(const uint64_t *base_addr, const size_t *base_addr_size, int num_base_addr)
{
    for (int ix = 0; ix < num_base_addr; ix++) {
        hx_InvalidateDCache_by_Addr((volatile void *)(uintptr_t)base_addr[ix], base_addr_size[ix]);
    }
}

#endif // #if EI_PORTING_HIMAX_WE2 == 1