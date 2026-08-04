/* The Clear BSD License
 *
 * Copyright (c) 2025 EdgeImpulse Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _EI_CLASSIFIER_INFERENCING_ENGINE_VLM_CONNECTOR_H_
#define _EI_CLASSIFIER_INFERENCING_ENGINE_VLM_CONNECTOR_H_

#if (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_VLM_CONNECTOR)

#include "model-parameters/model_metadata.h"
#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/porting/ei_logging.h"
#include <curl/curl.h>
#include <sstream>
#include <string>
#include "json/json.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <regex>
#include <base64/at_base64_lib.h>
#include "jpeg/encode_as_jpg.h"

enum class vlm_model_kind_t { CLIP, VLM, UNKNOWN };

static vlm_model_kind_t resolve_model_kind(const char* model) {
    if (strcmp(model, "clip-vit-large-patch14_ggml-model-q8_0") == 0) return vlm_model_kind_t::CLIP;
    if (strcmp(model, "Qwen3VL-8B-Instruct-Q8_0") == 0) return vlm_model_kind_t::VLM;
    return vlm_model_kind_t::UNKNOWN;
}

static std::vector<std::string> build_payloads(vlm_model_kind_t kind,
                                               const ei_impulse_t* impulse,
                                               const ei_learning_block_config_vlm_connection_t& block_config,
                                               uint8_t* image_base64,
                                               size_t size_out)
{
    const std::string image_url = "data:image/jpeg;base64," + std::string((char*)image_base64, size_out);
    std::vector<std::string> result;

    try {
        switch (kind) {
            case vlm_model_kind_t::CLIP: {
                // One payload for all description groups
                nlohmann::json payload;

                std::vector<std::string> descs;

                for (uint32_t i = 0; i < impulse->label_count; i++) {
                    descs.push_back(std::string(block_config.class_descriptions[i]));
                }

                payload["image_url"] = image_url;
                payload["class_descriptions"] = descs;
                result.push_back(payload.dump());
                break;
            }
            case vlm_model_kind_t::VLM: {
                nlohmann::json payload;
                payload["image_url"] = image_url;
                payload["model"] = block_config.model;
                payload["max_tokens"] = block_config.max_tokens;
                payload["temperature"] = block_config.temperature;
                // logit bias example (disable new line token)
                //payload["logit_bias"] = { "1734", -100 };
                if (block_config.response_schema != nullptr) {
                    payload["response_format"] = {
                        {"type", "json_schema"},
                        {"schema", nlohmann::json::parse(block_config.response_schema)}
                    };
                }
                payload["messages"] = {
                    {
                        {"role", "system"},
                        {"content", block_config.system_prompt}
                    },
                    {
                        {"role", "user"},
                        {"content", {
                            {
                                {"type", "image_url"},
                                {"image_url", {{"url", image_url}}}
                            },
                            {
                                {"type", "text"},
                                {"text", block_config.user_prompt}
                            }
                        }}
                    }
                };
                result.push_back(payload.dump());
                break;
            }
            default:
                break;
        }
    }
    catch (const std::exception& e) {
        EI_LOGE("Building JSON payload: %s\n", e.what());
        return {};
    }
    return result;
}

bool replace_all(std::string& input, const std::string& replace_word, const std::string& replace_by) {

    // Find the first occurrence of the substring
    size_t pos = input.find(replace_word);

    // Iterate through the string and replace all
    // occurrences
    while (pos != std::string::npos) {
        // Replace the substring with the specified string
        input.replace(pos, replace_word.size(), replace_by);

        // Find the next occurrence of the substring
        pos = input.find(replace_word,
                         pos + replace_by.size());
    }
    return true;
}

static std::string sanitize_vlm_content(std::string s) {
    replace_all(s, "```json", "");
    replace_all(s, "```", "");
    replace_all(s, "\\n", " ");
    s.erase(std::remove(s.begin(), s.end(), '\\'), s.end());
    if (!s.empty() && s.front() == '\"' && s.back() == '\"') {
        s = s.substr(1, s.size() - 2);
    }
    return s;
}

static EI_IMPULSE_ERROR parse_response(vlm_model_kind_t kind,
                                       const std::vector<std::string>& responses,
                                       ei_impulse_result_t* result,
                                       ei_learning_block_config_vlm_connection_t* block_config,
                                       const ei_impulse_t* impulse,
                                       uint32_t learn_block_index)
{
    switch (kind) {
        case vlm_model_kind_t::CLIP: {
            nlohmann::json json_result;
            try {
                json_result = nlohmann::json::parse(responses[0]);
            }
            catch (const std::exception& e) {
                EI_LOGE("Failed to parse CLIP response: %s\n", e.what());
                return EI_IMPULSE_INFERENCE_ERROR;
            }

            if (EI_LOG_LEVEL >= 5) {
                std::cout << "CLIP response JSON:\n";
                std::cout << std::setw(4) << json_result << '\n';
            }

            if (!json_result.contains("scores") || !json_result.contains("indices")) {
                EI_LOGE("CLIP response missing 'scores' or 'indices'\n");
                return EI_IMPULSE_INFERENCE_ERROR;
            }

            auto scores = json_result["scores"];
            auto indices = json_result["indices"];

            if (scores.size() != indices.size()) {
                EI_LOGE("CLIP response has mismatched scores/indices sizes\n");
                return EI_IMPULSE_INFERENCE_ERROR;
            }

            size_t num_labels = impulse->label_count;
            result->_raw_outputs[learn_block_index].matrix = new matrix_t(1, num_labels);
            for (size_t i = 0; i < num_labels; i++) {
                result->_raw_outputs[learn_block_index].matrix->buffer[i] = 0.0f;
            }

            for (size_t i = 0; i < scores.size(); i++) {
                uint32_t index = indices[i];
                if (index >= num_labels) {
                    EI_LOGE("CLIP response has out-of-range index %u (label_count %zu)\n", index, num_labels);
                    return EI_IMPULSE_INFERENCE_ERROR;
                }
                result->_raw_outputs[learn_block_index].matrix->buffer[index] = (float)scores[i];
            }
            break;
        }
        case vlm_model_kind_t::VLM: {
            nlohmann::json json_result;
            try {
                json_result = nlohmann::json::parse(responses[0]);
            }
            catch (const std::exception& e) {
                EI_LOGE("Failed to parse VLM response: %s\n", e.what());
                return EI_IMPULSE_INFERENCE_ERROR;
            }

            if (EI_LOG_LEVEL >= 5) {
                std::cout << "response JSON string: \n";
                std::cout << std::setw(4) << json_result << '\n';
            }

            std::string json_string = sanitize_vlm_content(
                json_result["choices"][0]["message"]["content"].dump());
            EI_LOGD("Final JSON String: %s\n", json_string.c_str());

            nlohmann::json json_response;
            try {
                json_response = nlohmann::json::parse(json_string);
            }
            catch (const std::exception& e) {
                EI_LOGE("Failed to parse VLM content JSON: %s\n", e.what());
                return EI_IMPULSE_INFERENCE_ERROR;
            }
            if (EI_LOG_LEVEL >= 5) {
                std::cout << std::setw(4) << json_response << '\n';
            }

            if (!json_response.is_object()) {
                EI_LOGE("JSON response is not an object.\n");
                return EI_IMPULSE_INFERENCE_ERROR;
            }

            result->_raw_outputs[learn_block_index].matrix = new matrix_t(1, impulse->label_count);
            for (size_t j = 0; j < impulse->label_count; j++) {
                result->_raw_outputs[learn_block_index].matrix->buffer[j] = 0.0f;
            }

            bool found_any_match = false;
            bool is_label_map = false;
            if (block_config->labeling_method != nullptr) {
                is_label_map = (strcmp(block_config->labeling_method, "label_map") == 0);
            }

            if (!is_label_map) {
                if (!json_response.contains("label") || !json_response["label"].is_string()) {
                    EI_LOGE("JSON response does not contain a valid 'label' field.\n");
                    return EI_IMPULSE_INFERENCE_ERROR;
                }
                std::string predicted_label = json_response["label"].get<std::string>();
                EI_LOGD("Parsed single-label response: label = '%s'\n", predicted_label.c_str());
                for (size_t j = 0; j < impulse->label_count; j++) {
                    if (impulse->categories[j] == predicted_label) {
                        result->_raw_outputs[learn_block_index].matrix->buffer[j] = 1.0f;
                        found_any_match = true;
                        break;
                    }
                }
                if (!found_any_match) {
                    EI_LOGE("No matching category found for label '%s'\n", predicted_label.c_str());
                    return EI_IMPULSE_INFERENCE_ERROR;
                }
            }
            else {
                // Multi-label mode: each string value in the response object is a predicted label
                for (auto it = json_response.begin(); it != json_response.end(); ++it) {
                    const std::string& response_key = it.key();
                    const nlohmann::json& response_value = it.value();
                    if (!response_value.is_string()) {
                        EI_LOGD("Skipping non-string response field '%s'\n", response_key.c_str());
                        continue;
                    }
                    std::string predicted_label = response_value.get<std::string>();
                    EI_LOGD("Parsed multi-label response field '%s' = '%s'\n",
                            response_key.c_str(), predicted_label.c_str());
                    bool matched = false;
                    for (size_t j = 0; j < impulse->label_count; j++) {
                        if (impulse->categories[j] == predicted_label) {
                            result->_raw_outputs[learn_block_index].matrix->buffer[j] = 1.0f;
                            matched = true;
                            found_any_match = true;
                            break;
                        }
                    }
                    if (!matched) {
                        EI_LOGD("No matching category found for response field '%s' value '%s'\n",
                                response_key.c_str(), predicted_label.c_str());
                    }
                }
                if (!found_any_match) {
                    EI_LOGE("JSON response did not contain any matching classification labels.\n");
                    return EI_IMPULSE_INFERENCE_ERROR;
                }
            }
            break;
        }
        default:
            return EI_IMPULSE_INFERENCE_ERROR;
    }

    result->_raw_outputs[learn_block_index].blockId = block_config->block_id;
    return EI_IMPULSE_OK;
}

EI_IMPULSE_ERROR check_response_code(CURLcode res, std::string& response) {
    if (res != CURLE_OK) {
        EI_LOGE("CURL error: %s\n", curl_easy_strerror(res));
        return EI_IMPULSE_INFERENCE_ERROR;
    }
    else {
        EI_LOGD(response.c_str());
        // Parse the JSON response
        nlohmann::json json_result;
        try {
            json_result = nlohmann::json::parse(response);
        }
        catch (const std::exception& e) {
            EI_LOGE("Failed to parse API response: %s\n", e.what());
            return EI_IMPULSE_INFERENCE_ERROR;
        }
        if (json_result.contains("error")) {
            std::string error_message = json_result["error"]["message"];
            EI_LOGE("API Error: %s\n", error_message.c_str());
            return EI_IMPULSE_INFERENCE_ERROR;
        }
    }
    return EI_IMPULSE_OK;
}

// Callback function to handle the response data
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::string* buffer = static_cast<std::string*>(userp);
    size_t total_size = size * nmemb;
    buffer->append(static_cast<char*>(contents), total_size);
    return total_size;
}

static EI_IMPULSE_ERROR perform_post(CURL* curl,
                                     const ei_learning_block_config_vlm_connection_t* block_config,
                                     const std::string& payload,
                                     std::string& response,
                                     uint64_t& classification_us)
{
    curl_easy_setopt(curl, CURLOPT_URL, block_config->server_url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)payload.size());

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Authorization: Bearer NONE");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    uint64_t req_start_us = ei_read_timer_us();
    CURLcode res = curl_easy_perform(curl);
    uint64_t req_end_us = ei_read_timer_us();

    curl_slist_free_all(headers);

    curl_off_t curl_total_time_us = 0;
    if (curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME_T, &curl_total_time_us) == CURLE_OK && curl_total_time_us > 0) {
        classification_us = static_cast<uint64_t>(curl_total_time_us);
    } else {
        classification_us = req_end_us - req_start_us;
    }

    return check_response_code(res, response);
}

EI_IMPULSE_ERROR run_vlm_inference(
    const ei_impulse_t *impulse,
    ei_feature_t *fmatrix,
    uint32_t learn_block_index,
    uint32_t* input_block_ids,
    uint32_t input_block_ids_size,
    ei_impulse_result_t *result,
    void *config_ptr,
    bool debug = false)
{
    // dummy function to match signature
    return EI_IMPULSE_OK;
}

EI_IMPULSE_ERROR run_vlm_inference(
    const ei_impulse_handle_t *handle,
    signal_t *signal,
    uint32_t learn_block_index,
    ei_impulse_result_t *result,
    void *config_ptr,
    bool debug = false)
{
    uint32_t input_width;
    uint32_t input_height;
    uint32_t nn_input_frame_size;
    const ei_impulse_t *impulse = handle->impulse;

    uint64_t dsp_start_us = ei_read_timer_us();

    if (handle->input_params) {
        ei_input_params *params = handle->input_params;
        input_width = params->input_width;
        input_height = params->input_height;
        nn_input_frame_size = params->nn_input_frame_size;
    }
    else {
        input_width = impulse->input_width;
        input_height = impulse->input_height;
        nn_input_frame_size = impulse->nn_input_frame_size;
    }

    EI_LOGD("Running VLM Connector inference (%ux%u)\r\n", input_width, input_height);
    ei_learning_block_config_vlm_connection_t *block_config = (ei_learning_block_config_vlm_connection_t*)config_ptr;

    vlm_model_kind_t kind = resolve_model_kind(block_config->model);
    if (kind == vlm_model_kind_t::UNKNOWN) {
        EI_LOGE("Unknown model type: %s\r\n", block_config->model);
        return EI_IMPULSE_INFERENCE_ERROR;
    }

    // allocate buffer for the jpeg image
    std::unique_ptr<uint8_t> image_data_ptr(new uint8_t[nn_input_frame_size]);
    uint8_t* image_data = image_data_ptr.get();
    if (image_data == nullptr) {
        EI_LOGE("Failed to allocate memory for image\r\n");
        return EI_IMPULSE_INFERENCE_ERROR;
    }

    size_t jpeg_size = 0;

    // convert signal to image
    int jpeg_ret = encode_rgb888_signal_as_jpg(
        signal,
        input_width,
        input_height,
        image_data,
        nn_input_frame_size,
        &jpeg_size);

    if (jpeg_ret != 0) {
        EI_LOGE("Failed to convert signal to jpeg image (%d)\r\n", jpeg_ret);
        return EI_IMPULSE_INFERENCE_ERROR;
    }

    // create output buffer (base64 needs size rounded up to multiple of 3)
    size_t output_size_check = ((jpeg_size + 2) / 3) * 4;

    std::unique_ptr<uint8_t[]> buffer_out_ptr(new uint8_t[output_size_check]);
    uint8_t* buffer_out = buffer_out_ptr.get();

    // convert image to base64
    size_t size_out = base64_encode_buffer((const char*)image_data,
                                    jpeg_size,
                                    (char*)buffer_out,
                                    output_size_check);

    if (size_out < 0) {
        EI_LOGE("Failed to convert image to base64 (%zu)\r\n", size_out);
        return EI_IMPULSE_INFERENCE_ERROR;
    }

    result->timing.dsp_us = ei_read_timer_us() - dsp_start_us;
    result->timing.dsp = (int)(result->timing.dsp_us / 1000);

    std::vector<std::string> payloads = build_payloads(kind, impulse, *block_config, buffer_out, size_out);
    if (payloads.empty()) {
        EI_LOGE("No payloads were generated\r\n");
        return EI_IMPULSE_INFERENCE_ERROR;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        EI_LOGE("Failed to initialize CURL\r\n");
        return EI_IMPULSE_INFERENCE_ERROR;
    }

    std::vector<std::string> responses;
    responses.reserve(payloads.size());
    uint64_t classification_total_us = 0;

    for (size_t group_ix = 0; group_ix < payloads.size(); ++group_ix) {
        std::string response;
        uint64_t req_us = 0;

        EI_LOGD("Sending payload %u of %u\r\n",
            static_cast<unsigned>(group_ix + 1),
            static_cast<unsigned>(payloads.size()));

        EI_IMPULSE_ERROR post_res = perform_post(curl, block_config, payloads[group_ix], response, req_us);
        classification_total_us += req_us;

        if (post_res != EI_IMPULSE_OK) {
            curl_easy_cleanup(curl);
            return post_res;
        }

        responses.emplace_back(std::move(response));
    }

    result->timing.classification_us = classification_total_us;
    result->timing.classification = static_cast<int>(classification_total_us / 1000);

    EI_LOGD("VLM Connector DSP time: %d ms\r\n", result->timing.dsp);
    EI_LOGD("VLM Connector DSP time: %lld us\r\n", result->timing.dsp_us);
    EI_LOGD("VLM Connector inference time: %d ms\r\n", result->timing.classification);
    EI_LOGD("VLM Connector inference time: %lld us\r\n", result->timing.classification_us);

    curl_easy_cleanup(curl);

    return parse_response(kind, responses, result, block_config, impulse, learn_block_index);
}

#endif // (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_VLM_CONNECTOR)
#endif // _EI_CLASSIFIER_INFERENCING_ENGINE_VLM_CONNECTOR_H_