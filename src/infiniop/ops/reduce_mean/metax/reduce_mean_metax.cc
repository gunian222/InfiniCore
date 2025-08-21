
#include "reduce_mean_metax.h"
#include <numeric>
#include "infiniop/devices/metax/metax_runtime.h"

namespace op::reduce_mean::metax {

Result<ReduceMeanInfo> ReduceMeanInfo::create(
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    const int* axes,
    int num_axes,
    int keep_dims) {
    ReduceMeanInfo info;
    info.axes = std::vector<int>(axes, axes + num_axes);
    info.num_axes = num_axes;
    info.keep_dims = keep_dims;
    info.dtype = input_desc->dtype;

    info.input_shape = std::vector<size_t>(
        input_desc->shape, input_desc->shape + input_desc->ndim);
    info.output_shape = std::vector<size_t>(
        output_desc->shape, output_desc->shape + output_desc->ndim);
    info.input_strides = std::vector<ptrdiff_t>(
        input_desc->strides, input_desc->strides + input_desc->ndim);
    info.output_strides = std::vector<ptrdiff_t>(
        output_desc->strides, output_desc->strides + output_desc->ndim);

    info.reduce_size = 1;
    for (int i = 0; i < num_axes; ++i) {
        int axis = axes[i];
        if (axis < 0) axis += input_desc->ndim;
        info.reduce_size *= info.input_shape[axis];
    }

    return info;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor**desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    const int* axes,
    int num_axes,
    int keep_dims) {
    auto info_result = ReduceMeanInfo::create(output_desc, input_desc, axes, num_axes, keep_dims);
    if (!info_result.ok()) return INFINI_STATUS_INVALID_ARGUMENT;

    *desc_ptr = new Descriptor(
        nullptr,
        info_result.value(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
INFINIOP_METAX_KERNEL void reduceMeanKernel(
    const T* input,
    T* output,
    ReduceMeanInfo info) {
    const size_t output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t output_size = std::accumulate(
        info.output_shape.begin(), info.output_shape.end(), 1, std::multiplies<size_t>());
    if (output_idx >= output_size) return;

    // 计算输入基础偏移量（复用MetaX工具函数）
    size_t input_offset = device::metax::indexToOffset(
        output_idx, info.output_shape.size(), info.output_shape.data(), info.input_strides.data());

    // 累加归约维度
    T sum = 0;
    for (size_t r = 0; r < info.reduce_size; ++r) {
        size_t reduce_offset = 0;
        size_t temp_r = r;
        for (int i = 0; i < info.num_axes; ++i) {
            int axis = info.axes[i];
            size_t dim_size = info.input_shape[axis];
            reduce_offset += (temp_r % dim_size) * info.input_strides[axis];
            temp_r /= dim_size;
        }
        sum += input[input_offset + reduce_offset];
    }

    // 计算平均值
    output[output_idx] = sum / static_cast<T>(info.reduce_size);
}

infiniStatus_t Descriptor::calculate(
    void* workspace,
    size_t workspace_size,
    void* output,
    const void* input,
    void* stream) const {
    const auto& info = _info;
    const size_t output_size = std::accumulate(
        info.output_shape.begin(), info.output_shape.end(), 1, std::multiplies<size_t>());

    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    switch (info.dtype) {
        case INFINI_DTYPE_F16:
            reduceMeanKernel<__half><<<grid_size, block_size, 0, (hc::accelerator_view*)stream>>>(
                (const __half*)input, (__half*)output, info);
            break;
        case INFINI_DTYPE_F32:
            reduceMeanKernel<float><<<grid_size, block_size, 0, (hc::accelerator_view*)stream>>>(
                (const float*)input, (float*)output, info);
            break;
        case INFINI_DTYPE_BF16:
            reduceMeanKernel<__hpcc_bfloat16><<<grid_size, block_size, 0, (hc::accelerator_view*)stream>>>(
                (const __hpcc_bfloat16*)input, (__hpcc_bfloat16*)output, info);
            break;
        default:
            return INFINI_STATUS_DATATYPE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}

}  // namespace op::reduce_mean::metax