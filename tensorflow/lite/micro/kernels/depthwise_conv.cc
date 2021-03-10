// Patched by Edge Impulse to include reference, CMSIS-NN and ARC kernels
#include "../../../../classifier/ei_classifier_config.h"
#if 0 == 1
/* noop */
#elif EI_CLASSIFIER_TFLITE_ENABLE_CMSIS_NN == 1
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "edge-impulse-sdk/CMSIS/NN/Include/arm_nnfunctions.h"
#include "edge-impulse-sdk/tensorflow/lite/c/builtin_op_data.h"
#include "edge-impulse-sdk/tensorflow/lite/c/common.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/common.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/quantization_util.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/kernel_util.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  // Scratch buffer for the arm_convolve_wrapper_s8 buffer
  void* scratch_buffer;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  int unused_output_height, unused_output_width;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      filter_height, filter_width, params->padding, &unused_output_height,
      &unused_output_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels));
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* raw;
  context->AllocatePersistentBuffer(context, sizeof(OpData), &raw);
  return raw;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if defined(__ARM_FEATURE_DSP) || defined(__ARM_FEATURE_MVE)
  OpData* data = static_cast<OpData*>(node->user_data);
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);

  const int filter_width = SizeOfDimension(filter, 2);
  const int filter_height = SizeOfDimension(filter, 1);

  RuntimeShape input_shape = GetTensorShape(input);
  const int input_depth = input_shape.Dims(3);

  // Dynamically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];

  void *per_channel_output_multiplier;
  context->AllocatePersistentBuffer(
      context, num_channels * sizeof(int32_t), &per_channel_output_multiplier);
  data->per_channel_output_multiplier =
      reinterpret_cast<int32_t*>(per_channel_output_multiplier);

  void *per_channel_output_shift;
  context->AllocatePersistentBuffer(
      context, num_channels * sizeof(int32_t), &per_channel_output_shift);
  data->per_channel_output_shift =
      reinterpret_cast<int32_t*>(per_channel_output_shift);

  data->scratch_buffer = nullptr;

  if (params->depth_multiplier == 1) {
    const int32_t buf_size = arm_depthwise_conv_s8_opt_get_buffer_size(
        input_depth, filter_width, filter_height);

    if (buf_size > 0) {
      context->AllocatePersistentBuffer(
          context, buf_size, &data->scratch_buffer);
    }
  }
#endif
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteDepthwiseConvParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  tflite::reference_ops::DepthwiseConv(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                                     TfLiteDepthwiseConvParams* params,
                                     OpData* data, const TfLiteTensor* input,
                                     const TfLiteTensor* filter,
                                     const TfLiteTensor* bias,
                                     TfLiteTensor* output) {
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output->params.zero_point;
  // TODO(b/130439627): Use calculated value for clamping.
  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

#if defined(__ARM_FEATURE_DSP) || defined(__ARM_FEATURE_MVE)
  RuntimeShape filter_shape = GetTensorShape(filter);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  RuntimeShape input_shape = GetTensorShape(input);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  RuntimeShape output_shape = GetTensorShape(output);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  RuntimeShape bias_shape = GetTensorShape(bias);

  if (op_params.depth_multiplier == 1) {
    int16_t* buf = reinterpret_cast<int16_t*>(data->scratch_buffer);

    TF_LITE_ENSURE_EQ(
        context,
        arm_depthwise_conv_s8_opt(
            GetTensorData<int8_t>(input), input_width, input_height,
            input_depth, GetTensorData<int8_t>(filter), input_depth,
            filter_width, filter_height, op_params.padding_values.width,
            op_params.padding_values.height, op_params.stride_width,
            op_params.stride_height, GetTensorData<int32>(bias),
            GetTensorData<int8_t>(output), data->per_channel_output_shift,
            data->per_channel_output_multiplier, output_width, output_height,
            op_params.output_offset, op_params.input_offset,
            op_params.quantized_activation_min,
            op_params.quantized_activation_max, op_params.dilation_width_factor,
            op_params.dilation_height_factor, buf),
        ARM_MATH_SUCCESS);
  } else {
    TF_LITE_ENSURE_EQ(
        context,
        arm_depthwise_conv_s8(
            GetTensorData<int8_t>(input), input_width, input_height,
            input_depth, GetTensorData<int8_t>(filter),
            op_params.depth_multiplier * input_depth,
            op_params.depth_multiplier, filter_width, filter_height,
            op_params.padding_values.width, op_params.padding_values.height,
            op_params.stride_width, op_params.stride_height,
            GetTensorData<int32>(bias), GetTensorData<int8_t>(output),
            data->per_channel_output_shift, data->per_channel_output_multiplier,
            output_width, output_height, op_params.output_offset,
            op_params.input_offset, op_params.quantized_activation_min,
            op_params.quantized_activation_max, op_params.dilation_width_factor,
            op_params.dilation_height_factor, nullptr),
        ARM_MATH_SUCCESS);
  }
#else
#pragma message( \
    "CMSIS-NN optimization for depthwise_conv not available for this target. Using reference kernel.")

  reference_integer_ops::DepthwiseConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8>(input), GetTensorShape(filter),
      GetTensorData<int8>(filter), GetTensorShape(bias),
      GetTensorData<int32>(bias), GetTensorShape(output),
      GetTensorData<int8>(output));

#endif
  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteDepthwiseConvParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data->output_shift;

#if defined(__ARM_FEATURE_DSP) || defined(__ARM_FEATURE_MVE)
  // optimizations utilize loop unrolling which requires the following power
  // of two kernel dimensions
  RuntimeShape filter_shape = GetTensorShape(filter);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  if (0 == op_params.depth_multiplier % 2 && 0 == filter_width % 2) {
    RuntimeShape input_shape = GetTensorShape(input);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    RuntimeShape output_shape = GetTensorShape(output);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    arm_depthwise_conv_u8_basic_ver1(
        GetTensorData<uint8_t>(input), input_width, input_height, input_depth,
        GetTensorData<uint8_t>(filter), filter_width, filter_height,
        op_params.depth_multiplier, op_params.padding_values.width,
        op_params.padding_values.height, op_params.stride_width,
        op_params.stride_height, op_params.dilation_width_factor,
        op_params.dilation_height_factor, GetTensorData<int32_t>(bias),
        op_params.input_offset, op_params.weights_offset,
        op_params.output_offset, GetTensorData<uint8_t>(output), output_width,
        output_height, op_params.quantized_activation_min,
        op_params.quantized_activation_max, op_params.output_shift,
        op_params.output_multiplier);
  } else
#endif

  {
    tflite::reference_ops::DepthwiseConv(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = static_cast<OpData*>(node->user_data);
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, data_type,
                                        data));

  // TODO(aselle): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat(context, node, params, data, input, filter, bias,
                       output);
      break;
    case kTfLiteInt8:
      return EvalQuantizedPerChannel(context, node, params, data, input,
                                     filter, bias, output);
      break;
    case kTfLiteUInt8:
      return EvalQuantized(context, node, params, data, input, filter, bias,
                           output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace depthwise_conv

TfLiteRegistration* Register_DEPTHWISE_CONV_2D() {
  static TfLiteRegistration r = {/*init=*/depthwise_conv::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/depthwise_conv::Prepare,
                                 /*invoke=*/depthwise_conv::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#elif EI_CLASSIFIER_TFLITE_ENABLE_ARC == 1
/* Copyright 2017-2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "mli_api.h"  // NOLINT
#include "edge-impulse-sdk/tensorflow/lite/c/builtin_op_data.h"
#include "edge-impulse-sdk/tensorflow/lite/c/common.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/common.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/quantization_util.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/kernel_util.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/padding.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/kernels/mli_slicers.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/kernels/mli_tf_utils.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/kernels/scratch_buf_mgr.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/kernels/scratch_buffers.h"

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

struct OpData {
  TfLitePaddingValues padding;

  // Cached tensor zero point values for quantized operations.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

bool IsMliApplicable(TfLiteContext* context, const TfLiteTensor* input,
                     const TfLiteTensor* filter, const TfLiteTensor* bias,
                     const TfLiteDepthwiseConvParams* params) {
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  const int in_ch = SizeOfDimension(input, 3);
  const int filters_num = SizeOfDimension(filter, 3);

  // MLI optimized version only supports int8_t dataype, dilation factor of 1
  // and per-axis quantization of weights (no broadcasting/per-tensor) (in_ch ==
  // filters_num) || (in_ch == 1)) is a forbidding of channel multiplier logic
  // for multichannel input.
  bool ret_val = (filter->type == kTfLiteInt8) &&
                 (input->type == kTfLiteInt8) && (bias->type == kTfLiteInt32) &&
                 (params->dilation_width_factor == 1) &&
                 (params->dilation_height_factor == 1) &&
                 (affine_quantization->scale->size ==
                  filter->dims->data[kDepthwiseConvQuantizedDimension]) &&
                 ((in_ch == filters_num) || (in_ch == 1));
  return ret_val;
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  int unused_output_height, unused_output_width;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      filter_height, filter_width, params->padding, &unused_output_height,
      &unused_output_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (data_type != kTfLiteFloat32 &&
      !IsMliApplicable(context, input, filter, bias, params)) {
    int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];

    return tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels);
  }
#endif
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* raw;
  context->AllocatePersistentBuffer(context, sizeof(OpData), &raw);
  return raw;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = static_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

  // Per channel quantization is only needed for int8 inference. For other
  // quantized types, only a single scale and zero point is needed.
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
  // Dynamically allocate per-channel quantization parameters.
  void* per_channel_output_multiplier;
  void* per_channel_output_shift;
  context->AllocatePersistentBuffer(context,
                                    num_channels * sizeof(int32_t),
                                    &per_channel_output_multiplier);
  context->AllocatePersistentBuffer(context,
                                    num_channels * sizeof(int32_t),
                                    &per_channel_output_shift);
  data->per_channel_output_multiplier = (int32_t*)per_channel_output_multiplier;
  data->per_channel_output_shift = (int32_t*)per_channel_output_shift;

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, data_type,
                                        data));

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  return kTfLiteOk;
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteDepthwiseConvParams* params, const OpData& data,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  tflite::reference_ops::DepthwiseConv(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
#else
  TF_LITE_KERNEL_LOG(context,
                     "Type %s (%d) is not supported by ARC MLI Library.",
                     TfLiteTypeGetName(input->type), input->type);
#endif
}
TfLiteStatus EvalMliQuantizedPerChannel(
    TfLiteContext* context, TfLiteNode* node, TfLiteDepthwiseConvParams* params,
    const OpData& data, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output) {
  // Run Depthwise Conv MLI kernel
  mli_tensor mli_in = {};
  mli_tensor mli_weights = {};
  mli_tensor mli_bias = {};
  mli_tensor mli_out = {};
  mli_conv2d_cfg cfg = {};

  // reuse space allocated for OpData parameters
  mli_weights.el_params.asym.scale.pi32 =
      (int32_t*)data.per_channel_output_multiplier;
  mli_bias.el_params.asym.scale.pi32 = (int32_t*)data.per_channel_output_shift;

  int16_t filter_zero_point = 0;
  int16_t bias_zero_point = 0;
  mli_weights.el_params.asym.zero_point.pi16 = &filter_zero_point;
  mli_bias.el_params.asym.zero_point.pi16 = &bias_zero_point;

  ConvertToMliTensor<int8_t>(input, &mli_in);
  ConvertToMliTensorPerChannel<int8_t>(filter, &mli_weights);
  ConvertToMliTensorPerChannel<int32_t>(bias, &mli_bias);
  ConvertToMliTensor<int8_t>(output, &mli_out);

  if (params->activation == kTfLiteActRelu) {
    cfg.relu.type = MLI_RELU_GEN;
  } else if (params->activation == kTfLiteActRelu6) {
    cfg.relu.type = MLI_RELU_6;
  } else if (params->activation == kTfLiteActRelu1) {
    cfg.relu.type = MLI_RELU_1;
  } else {
    cfg.relu.type = MLI_RELU_NONE;
  }

  cfg.stride_width = params->stride_width;
  cfg.stride_height = params->stride_height;
  if (params->padding == kTfLitePaddingValid) {
    cfg.padding_left = 0;
    cfg.padding_right = 0;
    cfg.padding_top = 0;
    cfg.padding_bottom = 0;
  } else {
    cfg.padding_left = data.padding.width;
    cfg.padding_right = data.padding.width + data.padding.width_offset;
    cfg.padding_top = data.padding.height;
    cfg.padding_bottom = data.padding.height + data.padding.height_offset;
  }

  // for height slicing
  const int heightDimension = 1;
  int inSliceHeight = 0;
  int outSliceHeight = 0;
  const int kernelHeight =
      static_cast<int>(mli_weights.shape[KRNL_DW_H_DIM_HWC]);
  const int overlap = kernelHeight - cfg.stride_height;

  // for weight slicing (on output channels)
  // HWCN layout for weigths, output channel dimension is the first dimension.
  const int weight_out_ch_dimension = 3;
  // bias has only 1 dimension
  const int bias_out_ch_dimension = 0;
  // Batch-Height-Width-Channel layout means last dimension is output channels.
  const int out_tensor_ch_dimension = 3;
  const int32_t in_channels = mli_in.shape[out_tensor_ch_dimension];
  const int32_t out_channels = mli_out.shape[out_tensor_ch_dimension];
  int slice_channels =
      static_cast<int>(mli_weights.shape[weight_out_ch_dimension]);

  // Tensors for data in fast (local) memory
  // and config to copy data from external to local memory
  mli_tensor weights_local = mli_weights;
  mli_tensor bias_local = mli_bias;
  mli_tensor in_local = mli_in;
  mli_tensor out_local = mli_out;  // this assumes that output shape
                                   // is already filled in the tensor struct.
  mli_mov_cfg_t copy_config;
  mli_mov_cfg_for_copy(&copy_config);

  TF_LITE_ENSURE_STATUS(get_arc_scratch_buffer_for_conv_tensors(
      context, &in_local, &weights_local, &bias_local, &out_local));
  /* is_local indicates that the tensor is already in local memory,
     so in that case the original tensor can be used,
     and there is no need to copy it to the local tensor*/
  const bool in_is_local = in_local.data == mli_in.data;
  const bool out_is_local = out_local.data == mli_out.data;
  const bool w_is_local = weights_local.data == mli_weights.data;
  const bool b_is_local = bias_local.data == mli_bias.data;

  TF_LITE_ENSURE_STATUS(arc_scratch_buffer_calc_slice_size_io(
      &in_local, &out_local, kernelHeight, cfg.stride_height, cfg.padding_top,
      cfg.padding_bottom, &inSliceHeight, &outSliceHeight));
  TF_LITE_ENSURE_STATUS(arc_scratch_buffer_calc_slice_size_weights(
      &weights_local, &bias_local, weight_out_ch_dimension, &slice_channels));

  /* if input channels is not equal to output channels, a channel multiplier
     is used. in this case the slice channels needs to be rounded down to a
     multiple of the input channels */
  if (in_channels != out_channels) {
    slice_channels = (slice_channels / in_channels) * in_channels;
  }

  TensorSlicer b_slice(&mli_bias, bias_out_ch_dimension, slice_channels);
  TensorSlicer w_slice(&mli_weights, weight_out_ch_dimension, slice_channels, 0,
                       0, 0, true);
  TensorSlicer out_ch_slice(&mli_out, out_tensor_ch_dimension, slice_channels,
                            0, 0, 0, true);
  TensorSlicer in_ch_slice(&mli_in, out_tensor_ch_dimension, slice_channels, 0,
                           0, 0, true);

  mli_tensor* w_ptr = w_is_local ? w_slice.Sub() : &weights_local;
  mli_tensor* b_ptr = b_is_local ? b_slice.Sub() : &bias_local;

  void* input_buffer_ptr = NULL;
  uint32_t input_buffer_size = 0;
  int padding_top = cfg.padding_top;
  int padding_bottom = cfg.padding_bottom;

  while (!w_slice.Done()) {
    mli_mov_tensor_sync(w_slice.Sub(), &copy_config, w_ptr);
    mli_mov_tensor_sync(b_slice.Sub(), &copy_config, b_ptr);

    /* input tensor is alreade sliced in the  channel dimension.
    out_ch_slice.Sub() is the tensor for the amount of channels of this
    itteration of the weight slice loop. This tensor needs to be further
    sliced over the batch and height dimension. in_ch_slice.Sub() tensor
    contains batches of HWC tensors. so it is a 4 dimensional tensor. because
    the mli kernel will process one HWC tensor at a time, the 4 dimensional
    tensor needs to be sliced into nBatch 3 dimensional tensors. on top of
    that there could be a need to also slice in the Height dimension. for that
    the sliceHeight has been calculated. The tensor slicer is configured that
    it will completely slice the nBatch dimension (0) and slice the height
    dimension (1) in chunks of 'sliceHeight' */
    TensorSlicer in_slice(in_ch_slice.Sub(), heightDimension, inSliceHeight,
                          padding_top, padding_bottom, overlap);

    /* output tensor is alreade sliced in the output channel dimension.
    out_ch_slice.Sub() is the tensor for the amount of output channels of this
    itteration of the weight slice loop. This tensor needs to be further
    sliced over the batch and height dimension. */
    TensorSlicer out_slice(out_ch_slice.Sub(), heightDimension, outSliceHeight);

    /* setup the pointers to the local or remote tensor to make the code
     * inside the loop easier. */
    mli_tensor* in_ptr = in_is_local ? in_slice.Sub() : &in_local;
    mli_tensor* out_ptr = out_is_local ? out_slice.Sub() : &out_local;

    while (!out_slice.Done()) {
      TF_LITE_ENSURE(context, !in_slice.Done());
      cfg.padding_top = in_slice.GetPaddingPre();
      cfg.padding_bottom = in_slice.GetPaddingPost();

      // if same input copy as previous iteration, skip the copy of input
      if ((in_slice.Sub()->data != input_buffer_ptr) ||
          (mli_hlp_count_elem_num(in_slice.Sub(), 0) != input_buffer_size)) {
        mli_mov_tensor_sync(in_slice.Sub(), &copy_config, in_ptr);
        input_buffer_ptr = in_slice.Sub()->data;
        input_buffer_size = mli_hlp_count_elem_num(in_slice.Sub(), 0);
      }
      mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32(in_ptr, w_ptr, b_ptr, &cfg,
                                                 out_ptr);
      mli_mov_tensor_sync(out_ptr, &copy_config, out_slice.Sub());

      in_slice.Next();
      out_slice.Next();
    }
    w_slice.Next();
    b_slice.Next();
    out_ch_slice.Next();
    in_ch_slice.Next();
    TF_LITE_ENSURE(context, in_slice.Done());
  }
  return kTfLiteOk;
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params,
                             const OpData& data, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.input_offset = -data.input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = data.output_zero_point;
  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  reference_integer_ops::DepthwiseConvPerChannel(
      op_params, data.per_channel_output_multiplier,
      data.per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8_t>(input), GetTensorShape(filter),
      GetTensorData<int8_t>(filter), GetTensorShape(bias),
      GetTensorData<int32_t>(bias), GetTensorShape(output),
      GetTensorData<int8_t>(output));
#else
  TF_LITE_KERNEL_LOG(context,
                     "Node configuration is not supported by ARC MLI Library.");
#endif
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteDepthwiseConvParams* params, const OpData& data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  const int32_t input_offset = -data.input_zero_point;
  const int32_t filter_offset = -data.filter_zero_point;
  const int32_t output_offset = data.output_zero_point;

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data.output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data.output_shift;

  tflite::reference_ops::DepthwiseConv(
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(filter), GetTensorData<uint8_t>(filter),
      GetTensorShape(bias), GetTensorData<int32_t>(bias),
      GetTensorShape(output), GetTensorData<uint8_t>(output));
#else
  TF_LITE_KERNEL_LOG(context,
                     "Type %s (%d) is not supported by ARC MLI Library.",
                     TfLiteTypeGetName(input->type), input->type);
#endif
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(context, node, params, data, input, filter, bias, output);
      break;
    case kTfLiteInt8:
      if (IsMliApplicable(context, input, filter, bias, params)) {
        EvalMliQuantizedPerChannel(context, node, params, data, input, filter,
                                   bias, output);
      } else {
        EvalQuantizedPerChannel(context, node, params, data, input, filter,
                                bias, output);
      }
      break;
    case kTfLiteUInt8:
      EvalQuantized(context, node, params, data, input, filter, bias, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace depthwise_conv

TfLiteRegistration* Register_DEPTHWISE_CONV_2D() {
  static TfLiteRegistration r = {/*init=*/depthwise_conv::Init,
          /*free=*/nullptr,
          /*prepare=*/depthwise_conv::Prepare,
          /*invoke=*/depthwise_conv::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#else
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "edge-impulse-sdk/tensorflow/lite/c/builtin_op_data.h"
#include "edge-impulse-sdk/tensorflow/lite/c/common.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/common.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/quantization_util.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/kernel_util.h"
#include "edge-impulse-sdk/tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  int unused_output_height, unused_output_width;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      filter_height, filter_width, params->padding, &unused_output_height,
      &unused_output_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];

    return tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels);
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = nullptr;
  if (context->AllocatePersistentBuffer(context, sizeof(OpData), &data) ==
      kTfLiteError) {
    return nullptr;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

  // Per channel quantization is only needed for int8 inference. For other
  // quantized types, only a single scale and zero point is needed.
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
  // Dynimically allocate per-channel quantization parameters.
  TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
      context, num_channels * sizeof(int32_t),
      reinterpret_cast<void**>(&data->per_channel_output_multiplier)));
  TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
      context, num_channels * sizeof(int32_t),
      reinterpret_cast<void**>(&data->per_channel_output_shift)));

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  return CalculateOpData(context, node, params, width, height, filter_width,
                         filter_height, data_type, data);
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteDepthwiseConvParams* params, const OpData* data,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  tflite::reference_ops::DepthwiseConv(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params,
                             const OpData* data, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output) {
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output->params.zero_point;
  // TODO(b/130439627): Use calculated value for clamping.
  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  reference_integer_ops::DepthwiseConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8>(input), GetTensorShape(filter),
      GetTensorData<int8>(filter), GetTensorShape(bias),
      GetTensorData<int32>(bias), GetTensorShape(output),
      GetTensorData<int8>(output));
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteDepthwiseConvParams* params, const OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data->output_shift;

  tflite::reference_ops::DepthwiseConv(
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(filter), GetTensorData<uint8_t>(filter),
      GetTensorShape(bias), GetTensorData<int32_t>(bias),
      GetTensorShape(output), GetTensorData<uint8_t>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  // TODO(aselle): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(context, node, params, &data, input, filter, bias, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel(context, node, params, &data, input, filter, bias,
                              output);
      break;
    case kTfLiteUInt8:
      EvalQuantized(context, node, params, &data, input, filter, bias, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace depthwise_conv

TfLiteRegistration* Register_DEPTHWISE_CONV_2D() {
  static TfLiteRegistration r = {/*init=*/depthwise_conv::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/depthwise_conv::Prepare,
                                 /*invoke=*/depthwise_conv::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif
