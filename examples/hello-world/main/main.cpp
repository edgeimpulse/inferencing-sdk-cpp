
#include "edge-impulse-sdk/classifier/ei_print_results.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model-parameters/model_metadata.h"
#include "offline_audio_sample.h"
#include "sdkconfig.h"

static const char* TAG = "hello-world";

static int16_t offline_audio[EI_CLASSIFIER_RAW_SAMPLE_COUNT]{};

static int get_signal_data(size_t offset, size_t length, float* out_ptr) {
    static uint32_t callback_counter = 0;

    if ((offset + length) > EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
        return -1;
    }

    callback_counter++;
    if ((callback_counter % 32) == 0) {
        vTaskDelay(1);
    }

    return ei::numpy::int16_to_float(&offline_audio[offset], out_ptr, length);
}

static void fill_offline_audio(void) {
    for (size_t i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
        offline_audio[i] = 0;
    }

    if (kOfflineKeywordSampleLength >= EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
        size_t start = (kOfflineKeywordSampleLength - EI_CLASSIFIER_RAW_SAMPLE_COUNT) / 2;
        for (size_t i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
            offline_audio[i] = kOfflineKeywordSample[start + i];
        }
    } else {
        size_t start = (EI_CLASSIFIER_RAW_SAMPLE_COUNT - kOfflineKeywordSampleLength) / 2;
        for (size_t i = 0; i < kOfflineKeywordSampleLength; i++) {
            offline_audio[start + i] = kOfflineKeywordSample[i];
        }
    }
}

extern "C" void app_main() {
#if CONFIG_EI_DISABLE_HW_ACCEL
    ESP_LOGW(TAG, "Hardware acceleration is disabled for this build. This may cause the classifier "
                  "to run significantly slower than expected.");
#endif

    fill_offline_audio();

    signal_t signal{};
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &get_signal_data;

    ei_impulse_result_t result{};

    ESP_LOGI(TAG, "Model: %s", EI_CLASSIFIER_PROJECT_NAME);
    ESP_LOGI(TAG, "Labels: %d", EI_CLASSIFIER_LABEL_COUNT);

    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    if (err != EI_IMPULSE_OK) {
        ESP_LOGE(TAG, "run_classifier failed (%d)", err);
    } else {
        ei_print_results(&ei_default_impulse, &result);
    }
}
