/*
 * timer.h
 *
 *  Created on: Jan 11, 2021
 *      Author: Naveen Kumar
 *
 * To get time in millis and micros since board has started
 * we need to rely on LP timer
 */

#ifndef PORTING_CY8CKIT_062_BLE_TIMER_H_
#define PORTING_CY8CKIT_062_BLE_TIMER_H_

#include "cyhal.h"
#include "cybsp.h"

cyhal_lptimer_t lptimerObj;
bool timer_initialized = false;

void timer_init() {
    cy_rslt_t result = cyhal_lptimer_init(&lptimerObj);

    /* LPTIMER initialization failed. Stop program execution */
    if (result == CY_RSLT_SUCCESS)
    {
    	timer_initialized = true;
    }
    else
    {
       /* Disable all interrupts */
        __disable_irq();

        /* Turn on LED to indicate error */
        cyhal_gpio_write(CYBSP_USER_LED, CYBSP_LED_STATE_ON);

        /* Halt the CPU */
        CY_ASSERT(0);
    }
}



#endif /* PORTING_CY8CKIT_062_BLE_TIMER_H_ */
