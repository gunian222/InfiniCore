
#ifndef __REDUCE_MEAN_NVIDIA_CUH__
#define __REDUCE_MEAN_NVIDIA_CUH__

#include "../../../operator.h"
#include "../../../tensor_descriptor.h"
#include "../reduce_mean.h"
#include "infiniop/devices/nvidia/cuda_common.h"

namespace op::reduce_mean::nvidia {

struct ReduceMeanInfo {
    std::vector<int> axes;
    int num_axes;
    int keep_dims;
    infiniDtype_t dtype;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    size_t reduce_size;  // 归约维度的总大小（用于计算平均值）

    static Result<ReduceMeanInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        const int* axes,
        int num_axes,
        int keep_dims);
};

class Descriptor : public op::Descriptor {
private:
    ReduceMeanInfo _info;

public:
    Descriptor(void* opaque, ReduceMeanInfo info, size_t workspace_size,
               infiniDevice_t device_type, int device_id)
        : op::Descriptor(opaque, workspace_size, device_type, device_id),
          _info(std::move(info)) {}

    ~Descriptor() override = default;

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor**desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        const int* axes,
        int num_axes,
        int keep_dims);

    infiniStatus_t calculate(
        void* workspace,
        size_t workspace_size,
        void* output,
        const void* input,
        void* stream) const override;

    size_t workspaceSize() const override { return 0; }  // 不需要工作区
};

}  // namespace op::reduce_mean::nvidia

#endif  // __REDUCE_MEAN_NVIDIA_CUH__