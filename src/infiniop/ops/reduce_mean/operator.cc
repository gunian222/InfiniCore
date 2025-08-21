#include "../../operator.h"  // 先包含结构体完整定义
#include "../../handle.h"
#include "infiniop/ops/reduce_mean.h"  // 后包含前向声明


struct InfiniopDescriptor;  // 冗余但安全的前向声明（可选）

// 后续使用 desc->device_type 时，编译器已识别完整定义#ifdef ENABLE_NVIDIA_API
#include "nvidia/reduce_mean_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "nvidia/reduce_mean_nvidia.cuh"  // 天数智芯复用NVIDIA实现
#endif
#ifdef ENABLE_METAX_API
#include "metax/reduce_mean_metax.h"
#endif

__C infiniStatus_t infiniopCreateReduceMeanDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMeanDescriptor_t* desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    const int* axes,
    int num_axes,
    int keep_dims) {

#define CREATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                \
        return op::reduce_mean::NAMESPACE::Descriptor::create(                \
            handle,                                                           \
            reinterpret_cast<op::reduce_mean::NAMESPACE::Descriptor**>(desc_ptr), \
            output_desc, input_desc, axes, num_axes, keep_dims)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);  // 天数智芯复用NVIDIA实现
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t infiniopGetReduceMeanWorkspaceSize(
    infiniopReduceMeanDescriptor_t desc, size_t* size) {

#define GET(CASE, NAMESPACE)                                                  \
    case CASE:                                                                \
        *size = reinterpret_cast<op::reduce_mean::NAMESPACE::Descriptor*>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopReduceMean(
    infiniopReduceMeanDescriptor_t desc,
    void* workspace,
    size_t workspace_size,
    void* output,
    const void* input,
    void* stream) {

#define CALCULATE(CASE, NAMESPACE)                                            \
    case CASE:                                                                \
        return reinterpret_cast<op::reduce_mean::NAMESPACE::Descriptor*>(desc)->calculate( \
            workspace, workspace_size, output, input, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyReduceMeanDescriptor(
    infiniopReduceMeanDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                               \
    case CASE:                                                                \
        delete reinterpret_cast<op::reduce_mean::NAMESPACE::Descriptor*>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}