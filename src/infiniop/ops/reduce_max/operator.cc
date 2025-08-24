#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/reduce_max.h"

#include "nvidia/reduce_max_nvidia.cuh"

__C infiniStatus_t infiniopCreateReduceMaxDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    const int* axes,
    int num_axes,
    int keep_dims) 
{
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return op::reduce_max::nvidia::Descriptor::create(
            handle,
            reinterpret_cast<op::reduce_max::nvidia::Descriptor **>(desc_ptr),
            y_desc,
            x_desc,
            axes,
            num_axes,
            keep_dims);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopGetReduceMaxWorkspaceSize(
    infiniopReduceMaxDescriptor_t desc, size_t *size) 
{
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        *size = reinterpret_cast<op::reduce_max::nvidia::Descriptor *>(desc)->workspaceSize();
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopReduceMax(
    infiniopReduceMaxDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) 
{
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return reinterpret_cast<op::reduce_max::nvidia::Descriptor *>(desc)->calculate(
            workspace, workspace_size, y, x, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopDestroyReduceMaxDescriptor(
    infiniopReduceMaxDescriptor_t desc) 
{
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        delete reinterpret_cast<op::reduce_max::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
