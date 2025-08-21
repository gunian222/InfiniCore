#include "reduce_mean_nvidia.cuh"
#include <numeric>

namespace op::reduce_mean::nvidia {

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

    // 复制输入/输出形状和步长
    info.input_shape = std::vector<size_t>(
        input_desc->shape, input_desc->shape + input_desc->ndim);
    info.output_shape = std::vector<size_t>(
        output_desc->shape, output_desc->shape + output_desc->ndim);
    info.input_strides = std::vector<ptrdiff_t>(
        input_desc->strides, input_desc->strides + input_desc->ndim);
    info.output_strides = std::vector<ptrdiff_t>(
        output_desc->strides, output_desc->strides + output_desc->ndim);

    // 计算归约维度的总大小（用于平均值计算）
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
        0,  // 无工作区
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
__global__ void reduceMeanKernel(
    const T* input,
    T* output,
    const ReduceMeanInfo info) {
    const size_t output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= std::accumulate(info.output_shape.begin(), info.output_shape.end(), 1, std::multiplies<size_t>())) {
        return;
    }

    // 计算输出索引对应的输入基础偏移量（非归约维度）
    size_t input_offset = 0;
    size_t temp_idx = output_idx;
    for (int i = info.output_shape.size() - 1; i >= 0; --i) {
        const size_t dim_size = info.output_shape[i];
        const size_t coord = temp_idx % dim_size;
        temp_idx /= dim_size;
        input_offset += coord * info.input_strides[i];
    }

    // 累加归约维度的值
    T sum = 0;
    const size_t reduce_size = info.reduce_size;
    for (size_t r = 0; r < reduce_size; ++r) {
        // 计算归约维度的偏移量
        size_t reduce_offset = 0;
        size_t temp_r = r;
        for (int i = 0; i < info.num_axes; ++i) {
            const int axis = info.axes[i];
            const size_t dim_size = info.input_shape[axis];
            const size_t coord = temp_r % dim_size;
            temp_r /= dim_size;
            reduce_offset += coord * info.input_strides[axis];
        }
        sum += input[input_offset + reduce_offset];
    }

    // 计算平均值并写入输出
    output[output_idx] = sum / static_cast<T>(reduce_size);
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

    // 启动核函数
    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    switch (info.dtype) {
        case INFINI_DTYPE_F16:
            reduceMeanKernel<half><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
                (const half*)input, (half*)output, info);
            break;
        case INFINI_DTYPE_F32:
            reduceMeanKernel<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
                (const float*)input, (float*)output, info);
            break;
        case INFINI_DTYPE_BF16:
            reduceMeanKernel<__nv_bfloat16><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
                (const __nv_bfloat16*)input, (__nv_bfloat16*)output, info);
            break;
        default:
            return INFINI_STATUS_DATATYPE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}

}  // namespace op::reduce_mean::nvidia