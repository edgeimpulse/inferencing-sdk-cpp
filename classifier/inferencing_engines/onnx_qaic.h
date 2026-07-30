/* The Clear BSD License
 *
 * Copyright (c) 2026 EdgeImpulse Inc.
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

#ifndef _EI_CLASSIFIER_INFERENCING_ENGINE_QAIC_H_
#define _EI_CLASSIFIER_INFERENCING_ENGINE_QAIC_H_

#if (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_QAIC)

#include "model-parameters/model_metadata.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/qaic/qaic_provider_factory.h>

#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/classifier/ei_model_types.h"

typedef struct {
    bool initialized;
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names_storage;
    std::vector<std::string> output_names_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<ONNXTensorElementDataType> input_types;
} ei_qaic_context_t;

static ei_qaic_context_t g_qaic_ctx = {};

static std::string qaic_project_path(const char *project_name,
                                     uint32_t project_id,
                                     uint16_t deploy_version)
{
    return "/tmp/" + std::string(project_name) + "-" +
        std::to_string(project_id) + "-" +
        std::to_string(deploy_version);
}

static EI_IMPULSE_ERROR write_qaic_runtime_config(const std::string &config_path, const ei_qaic_model_config_t *graph_config)
{
    FILE *file = fopen(config_path.c_str(), "w");
    if (file == nullptr) {
        ei_printf("ERR: Failed to create QAIC runtime config: %s\n", config_path.c_str());
        return EI_IMPULSE_DEVICE_INIT_ERROR;
    }

    if (graph_config->onnx_define_symbol != nullptr && graph_config->onnx_define_symbol[0] != '\0') {
        fprintf(file, "onnx-define-symbol: \"%s\"\n", graph_config->onnx_define_symbol);
    }
    fprintf(file, "aic-num-cores: %u\n", graph_config->aic_num_cores);
    fprintf(file, "aic-hw: %s\n", graph_config->aic_hw ? "True" : "False");
    fprintf(file, "convert-to-fp16: %s\n", graph_config->convert_to_fp16 ? "True" : "False");

    fclose(file);
    return EI_IMPULSE_OK;
}

static EI_IMPULSE_ERROR write_qaic_model_file(const std::string &project_path, const std::string &model_path, const ei_qaic_model_config_t *graph_config)
{
    if (mkdir(project_path.c_str(), 0777) != 0 && errno != EEXIST) {
        ei_printf("ERR: Failed to create QAIC model directory: %s\n", project_path.c_str());
        return EI_IMPULSE_DEVICE_INIT_ERROR;
    }

    FILE *file = fopen(model_path.c_str(), "wb");
    if (file == nullptr) {
        ei_printf("ERR: Failed to create QAIC model file: %s\n", model_path.c_str());
        return EI_IMPULSE_DEVICE_INIT_ERROR;
    }

    const size_t written = fwrite(graph_config->model, 1, graph_config->model_size, file);
    fclose(file);

    if (written != graph_config->model_size) {
        ei_printf("ERR: Failed to write QAIC model file: %s\n", model_path.c_str());
        return EI_IMPULSE_DEVICE_INIT_ERROR;
    }

    return EI_IMPULSE_OK;
}

static EI_IMPULSE_ERROR inference_qaic_setup(const char *project_name,
                                             uint32_t project_id,
                                             uint16_t deploy_version,
                                             const ei_qaic_model_config_t *graph_config)
{
    if (g_qaic_ctx.initialized) {
        return EI_IMPULSE_OK;
    }

    const std::string project_path = qaic_project_path(project_name, project_id, deploy_version);
    const std::string model_path = project_path + "/" + std::string(graph_config->model_filename);

    EI_IMPULSE_ERROR model_res = write_qaic_model_file(project_path, model_path, graph_config);
    if (model_res != EI_IMPULSE_OK) {
        return model_res;
    }

    g_qaic_ctx.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "edge-impulse-qaic");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const char *qaic_config_path = std::getenv("EI_QAIC_CONFIG_PATH");
    std::string runtime_config_path;
    if (qaic_config_path == nullptr) {
        runtime_config_path = project_path + "/qaic-runtime-config.yaml";
        EI_IMPULSE_ERROR config_res = write_qaic_runtime_config(runtime_config_path, graph_config);
        if (config_res != EI_IMPULSE_OK) {
            return config_res;
        }
        qaic_config_path = runtime_config_path.c_str();
    }

    auto ort_status = OrtSessionOptionsAppendExecutionProvider_QAic(
        session_options,
        qaic_config_path,
        graph_config->device_id);

    if (ort_status != nullptr) {
        ei_printf("ERR: Failed to append QAIC execution provider: %s\n",
            Ort::GetApi().GetErrorMessage(ort_status));
        Ort::GetApi().ReleaseStatus(ort_status);
        return EI_IMPULSE_DEVICE_INIT_ERROR;
    }

    try {
        // use unique pointer to avoid deallocating manually
        g_qaic_ctx.session = std::make_unique<Ort::Session>(*g_qaic_ctx.env, model_path.c_str(), session_options);
    }
    catch (const Ort::Exception& e) {
        ei_printf("ERR: Failed to create ONNX Runtime session for QAIC: %s\n", e.what());
        return EI_IMPULSE_ONNX_ERROR;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = g_qaic_ctx.session->GetInputCount();
    const size_t output_count = g_qaic_ctx.session->GetOutputCount();

    g_qaic_ctx.input_names_storage.reserve(input_count);
    g_qaic_ctx.input_names.reserve(input_count);
    g_qaic_ctx.input_shapes.reserve(input_count);
    g_qaic_ctx.input_types.reserve(input_count);

    for (size_t ix = 0; ix < input_count; ix++) {
#if ORT_API_VERSION >= 12
        auto name = g_qaic_ctx.session->GetInputNameAllocated(ix, allocator);
        g_qaic_ctx.input_names_storage.emplace_back(name.get());
#else
        char *name = g_qaic_ctx.session->GetInputName(ix, allocator);
        g_qaic_ctx.input_names_storage.emplace_back(name);
        allocator.Free(name);
#endif
        g_qaic_ctx.input_names.push_back(g_qaic_ctx.input_names_storage.back().c_str());

        auto type_info = g_qaic_ctx.session->GetInputTypeInfo(ix);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        g_qaic_ctx.input_shapes.emplace_back(tensor_info.GetShape());
        g_qaic_ctx.input_types.push_back(tensor_info.GetElementType());
    }

    g_qaic_ctx.output_names_storage.reserve(output_count);
    g_qaic_ctx.output_names.reserve(output_count);

    for (size_t ix = 0; ix < output_count; ix++) {
#if ORT_API_VERSION >= 12
        auto name = g_qaic_ctx.session->GetOutputNameAllocated(ix, allocator);
        g_qaic_ctx.output_names_storage.emplace_back(name.get());
#else
        char *name = g_qaic_ctx.session->GetOutputName(ix, allocator);
        g_qaic_ctx.output_names_storage.emplace_back(name);
        allocator.Free(name);
#endif
        g_qaic_ctx.output_names.push_back(g_qaic_ctx.output_names_storage.back().c_str());
    }

    g_qaic_ctx.initialized = true;
    return EI_IMPULSE_OK;
}

static EI_IMPULSE_ERROR fill_float_input(
    const ei_impulse_t *impulse,
    ei_feature_t *fmatrix,
    uint32_t *input_block_ids,
    uint32_t input_block_ids_size,
    std::vector<float> *input_values)
{
#if EI_CLASSIFIER_SINGLE_FEATURE_INPUT == 0
    const size_t mtx_size = impulse->dsp_blocks_size + impulse->learning_blocks_size;
    ei::matrix_t* matrix = nullptr;
    const size_t combined_matrix_size = get_feature_size(fmatrix, input_block_ids_size, input_block_ids, mtx_size);
    input_values->reserve(combined_matrix_size);

    for (size_t ix = 0; ix < input_block_ids_size; ix++) {
        const size_t cur_mtx = input_block_ids[ix];
        if (!find_mtx_by_idx(fmatrix, &matrix, cur_mtx, mtx_size)) {
            ei_printf("ERR: Cannot find matrix with id %zu\n", cur_mtx);
            return EI_IMPULSE_INVALID_SIZE;
        }

        for (size_t jx = 0; jx < matrix->rows * matrix->cols; jx++) {
            input_values->push_back(matrix->buffer[jx]);
        }
    }
#else
    const ei::matrix_t* matrix = fmatrix[0].matrix;
    input_values->assign(matrix->buffer, matrix->buffer + (matrix->rows * matrix->cols));
#endif

    return EI_IMPULSE_OK;
}

EI_IMPULSE_ERROR run_nn_inference(
    const ei_impulse_t *impulse,
    ei_feature_t *fmatrix,
    uint32_t learn_block_index,
    uint32_t* input_block_ids,
    uint32_t input_block_ids_size,
    ei_impulse_result_t *result,
    void *config_ptr,
    bool debug = false)
{
    (void)debug; // avoid unused variable
    ei_learning_block_config_tflite_graph_t *block_config = (ei_learning_block_config_tflite_graph_t*)config_ptr;
    ei_qaic_model_config_t *graph_config = (ei_qaic_model_config_t*)block_config->graph_config;

    if (graph_config == nullptr || graph_config->model == nullptr || graph_config->model_size == 0 ||
        graph_config->model_filename == nullptr || graph_config->model_filename[0] == '\0') {
        ei_printf("ERR: QAIC graph configuration is missing model data.\n");
        return EI_IMPULSE_INVALID_SIZE;
    }

    EI_IMPULSE_ERROR init_res = inference_qaic_setup(impulse->project_name, impulse->project_id, impulse->deploy_version, graph_config);
    if (init_res != EI_IMPULSE_OK) {
        return init_res;
    }

    if (g_qaic_ctx.input_names.size() != 1 || g_qaic_ctx.input_shapes.size() != 1 || g_qaic_ctx.input_types.size() != 1) {
        ei_printf("ERR: QAIC engine currently supports a single input model.\n");
        return EI_IMPULSE_ONNX_ERROR;
    }

    std::vector<float> input_values;
    EI_IMPULSE_ERROR fill_res = fill_float_input(impulse, fmatrix, input_block_ids, input_block_ids_size, &input_values);
    if (fill_res != EI_IMPULSE_OK) {
        return fill_res;
    }

    std::vector<int64_t> input_shape = g_qaic_ctx.input_shapes[0];
    for (size_t ix = 0; ix < input_shape.size(); ix++) {
        if (input_shape[ix] < 0) {
            input_shape[ix] = 1;
        }
    }

    size_t expected_elements = 1;
    for (size_t ix = 0; ix < input_shape.size(); ix++) {
        expected_elements *= static_cast<size_t>(input_shape[ix]);
    }
    if (expected_elements != input_values.size()) {
        ei_printf("ERR: Invalid input size, expected %zu values, got %zu\n", expected_elements, input_values.size());
        return EI_IMPULSE_INVALID_SIZE;
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_values.data(),
        input_values.size(),
        input_shape.data(),
        input_shape.size());

    uint64_t ctx_start_us = ei_read_timer_us();
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = g_qaic_ctx.session->Run(
            Ort::RunOptions{nullptr},
            g_qaic_ctx.input_names.data(),
            &input_tensor,
            1,
            g_qaic_ctx.output_names.data(),
            g_qaic_ctx.output_names.size());
    }
    catch (const Ort::Exception& e) {
        ei_printf("ERR: QAIC ONNX Runtime inference failed: %s\n", e.what());
        return EI_IMPULSE_ONNX_ERROR;
    }
    uint64_t ctx_end_us = ei_read_timer_us();
    result->timing.classification_us = ctx_end_us - ctx_start_us;

    for (size_t out_ix = 0; out_ix < output_tensors.size(); out_ix++) {
        auto tensor_info = output_tensors[out_ix].GetTensorTypeAndShapeInfo();
        const auto output_shape = tensor_info.GetShape();
        size_t output_size = 1;
        for (size_t dim_ix = 0; dim_ix < output_shape.size(); dim_ix++) {
            output_size *= static_cast<size_t>(std::max<int64_t>(output_shape[dim_ix], 1));
        }

        result->_raw_outputs[learn_block_index + out_ix].matrix = new matrix_t(1, output_size);
        result->_raw_outputs[learn_block_index + out_ix].blockId = block_config->block_id + out_ix;

        ONNXTensorElementDataType output_type = tensor_info.GetElementType();
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* out_data = output_tensors[out_ix].GetTensorData<float>();
            for (size_t val_ix = 0; val_ix < output_size; val_ix++) {
                result->_raw_outputs[learn_block_index + out_ix].matrix->buffer[val_ix] = out_data[val_ix];
            }
        }
        else {
            ei_printf("ERR: Unsupported QAIC output tensor type (%d)\n", (int)output_type);
            return EI_IMPULSE_ONNX_ERROR;
        }
    }

    return EI_IMPULSE_OK;
}

#endif // (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_QAIC) && (EI_CLASSIFIER_COMPILED != 1)
#endif // _EI_CLASSIFIER_INFERENCING_ENGINE_QAIC_H_
