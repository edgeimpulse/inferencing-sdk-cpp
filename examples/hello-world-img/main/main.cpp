/* Edge Impulse Espressif ESP32 Standalone Inference ESP IDF Example
 * Copyright (c) 2022 EdgeImpulse Inc.
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

/* Include ----------------------------------------------------------------- */
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "esp_heap_caps.h"
#include "offline_sample.h"
#include "sdkconfig.h"

#include <stdio.h>
#include <string.h>

int raw_feature_get_data(size_t offset, size_t length, float* out_ptr) {
    const size_t feature_count = sizeof(features) / sizeof(features[0]);
    if (offset + length > feature_count) {
        return -1;
    }

    for (size_t index = 0; index < length; index++) {
        out_ptr[index] = static_cast<float>(features[offset + index]);
    }
    return 0;
}

void print_inference_result(ei_impulse_result_t result) {
    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n", result.timing.dsp,
              result.timing.classification, result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n", bb.label, bb.value, bb.x,
                  bb.y, bb.width, bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    uint16_t top_index = 0;
    float top_value = result.classification[0].value;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > top_value) {
            top_value = result.classification[i].value;
            top_index = i;
        }
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }

    const char* expected_label = "plant";
    const char* predicted_label = ei_classifier_inferencing_categories[top_index];
    ei_printf("Top prediction: %s (%.5f)\r\n", predicted_label, top_value);
    ei_printf("Expected label: %s -> %s\r\n", expected_label,
              strcmp(predicted_label, expected_label) == 0 ? "PASS" : "FAIL");
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif
}

void print_memory_diagnostics() {
    size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    size_t internal_largest =
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    ei_printf("Heap (internal): free=%u, largest=%u bytes\r\n",
              static_cast<unsigned int>(internal_free),
              static_cast<unsigned int>(internal_largest));

#if CONFIG_SPIRAM
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    size_t psram_largest = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    ei_printf("PSRAM: enabled, free=%u, largest=%u bytes\r\n",
              static_cast<unsigned int>(psram_free), static_cast<unsigned int>(psram_largest));
#else
    ei_printf("PSRAM: disabled in sdkconfig (CONFIG_SPIRAM=n)\r\n");
#endif

    ei_printf("Model arena target: %u bytes\r\n",
              static_cast<unsigned int>(EI_CLASSIFIER_TFLITE_LARGEST_ARENA_SIZE));
}

extern "C" int app_main() {
    ei_sleep(100);

    ei_impulse_result_t result = {nullptr};

    ei_printf("Edge Impulse standalone inferencing (Espressif ESP32)\n");
    print_memory_diagnostics();

    if (sizeof(features) / sizeof(features[0]) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf(
            "The size of your 'features' array is not correct. Expected %d items, but had %u\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(features[0]));
        return 1;
    }

    while (true) {
        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
        if (res != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", res);
            return res;
        }

        print_inference_result(result);

        ei_sleep(1000);
    }
}
