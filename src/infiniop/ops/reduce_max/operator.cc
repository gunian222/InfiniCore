#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/reduce_max.h"

// 英伟达平台（含兼容架构）
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/reduce_max_nvidia.cuh"
#endif

// 天数智芯平台
#ifdef ENABLE_TIANSHU_API
#include "tianshu/reduce_max_tianshu.h"
#endif

// 沐曦平台
#ifdef ENABLE_MOXI_API
#include "moxi/reduce_max_moxi.h"
#endif

__C infiniStatus_t infiniopCreateReduceMaxDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    const int *axes,
    int num_axes,
    int keep_dims) {

#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::reduce_max::NAMESPACE::Descriptor::create(                 \
            handle,                                                            \
            reinterpret_cast<op::reduce_max::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                       \
            input_desc,                                                        \
            axes,                                                              \
            num_axes,                                                          \
            keep_dims)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia); // 复用英伟达实现
#endif
#ifdef ENABLE_TIANSHU_API
        CREATE(INFINI_DEVICE_TIANSHU, tianshu);
#endif
#ifdef ENABLE_MOXI_API
        CREATE(INFINI_DEVICE_MOXI, moxi);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetReduceMaxWorkspaceSize(
    infiniopReduceMaxDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                               \
    case CASE:                                                                             \
        *size = reinterpret_cast<op::reduce_max::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_TIANSHU_API
        GET(INFINI_DEVICE_TIANSHU, tianshu);
#endif
#ifdef ENABLE_MOXI_API
        GET(INFINI_DEVICE_MOXI, moxi);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopReduceMax(
    infiniopReduceMaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                   \
        return reinterpret_cast<const op::reduce_max::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_TIANSHU_API
        CALCULATE(INFINI_DEVICE_TIANSHU, tianshu);
#endif
#ifdef ENABLE_MOXI_API
        CALCULATE(INFINI_DEVICE_MOXI, moxi);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyReduceMaxDescriptor(
    infiniopReduceMaxDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        delete reinterpret_cast<const op::reduce_max::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_TIANSHU_API
        DELETE(INFINI_DEVICE_TIANSHU, tianshu);
#endif
#ifdef ENABLE_MOXI_API
        DELETE(INFINI_DEVICE_MOXI, moxi);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}