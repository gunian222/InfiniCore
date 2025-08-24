#include "../../../devices/nvidia/nvidia_common.cuh"
#include "reduce_max_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>              // 如已有 reduce 基元用到 CUB，可保留
#include "../../../reduce/cuda/reduce.cuh"         // 你工程里已有：op::common_cuda::reduce_op::max
#include "../cuda/kernel.cuh"                      // 若你的工程中存在公共 kernel 依赖，可保留



namespace op::reduce_max::nvidia {

// ---- Descriptor Opaque ----
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

// 假设你已有 ReduceMaxInfo::create(...)，给出 dtype/shape/stride
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc)
{
    auto info = ReduceMaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), /*workspace_size=*/0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// ---- kernel 启动模板 ----
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, infiniDtype_t dtype,
                            size_t batch, size_t height, size_t width,
                            ptrdiff_t y_stride_b, ptrdiff_t y_stride_h,
                            ptrdiff_t x_stride_b, ptrdiff_t x_stride_h,
                            cudaStream_t stream)
{
    // 每个 block 负责一行：grid=(height, batch)
    dim3 grid(static_cast<uint32_t>(height), static_cast<uint32_t>(batch), 1);

    if (dtype == INFINI_DTYPE_F16) {
        reduceMax<BLOCK_SIZE, half>
            <<<grid, BLOCK_SIZE, 0, stream>>>(
                static_cast<half*>(y), static_cast<const half*>(x),
                batch, height, width,
                y_stride_b, y_stride_h,
                x_stride_b, x_stride_h);
    } else if (dtype == INFINI_DTYPE_BF16) {
        reduceMax<BLOCK_SIZE, __nv_bfloat16>
            <<<grid, BLOCK_SIZE, 0, stream>>>(
                static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x),
                batch, height, width,
                y_stride_b, y_stride_h,
                x_stride_b, x_stride_h);
    } else if (dtype == INFINI_DTYPE_F32) {
        reduceMax<BLOCK_SIZE, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>(
                static_cast<float*>(y), static_cast<const float*>(x),
                batch, height, width,
                y_stride_b, y_stride_h,
                x_stride_b, x_stride_h);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y, const void *x, void *stream_) const
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);


    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            y, x, _info.dtype, _info.batch_size, _info.height, _info.width,
            _info.y_stride_b, _info.y_stride_h,
            _info.x_stride_b, _info.x_stride_h, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            y, x, _info.dtype, _info.batch_size, _info.height, _info.width,
            _info.y_stride_b, _info.y_stride_h,
            _info.x_stride_b, _info.x_stride_h, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            y, x, _info.dtype, _info.batch_size, _info.height, _info.width,
            _info.y_stride_b, _info.y_stride_h,
            _info.x_stride_b, _info.x_stride_h, stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::reduce_max::nvidia
