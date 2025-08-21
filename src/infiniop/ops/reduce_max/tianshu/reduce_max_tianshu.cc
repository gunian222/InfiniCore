#include "reduce_max_tianshu.h"
#include "../../../device/tianshu/handle.h"
#include "../../../tensor/tensor.h"

namespace op::reduce_max::tianshu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    const int *axes,
    int num_axes,
    int keep_dims) {

    auto handle = reinterpret_cast<device::tianshu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // 天数平台支持的数据类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // 验证输入输出形状
    CHECK_REDUCE_SHAPE(input_desc->shape(), out_desc->shape(), axes, num_axes, keep_dims);

    auto desc = new Descriptor();
    desc->_handle = handle;
    desc->_dtype = dtype;
    desc->_input_desc = input_desc;
    desc->_output_desc = out_desc;
    desc->_axes = std::vector<int>(axes, axes + num_axes);
    desc->_keep_dims = keep_dims;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    const auto &input_shape = _input_desc->shape();
    const auto &output_shape = _output_desc->shape();
    const auto &input_strides = _input_desc->strides();
    const auto &output_strides = _output_desc->strides();

    // 调度天数平台核函数
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return reduce::tianshu::launch_kernel<ReduceMaxOp, fp16_t>(
            _handle,
            ReduceMaxOp(),
            output,
            input,
            input_shape,
            output_shape,
            input_strides,
            output_strides,
            _axes,
            _keep_dims,
            workspace,
            workspace_size,
            stream);
    case INFINI_DTYPE_F32:
        return reduce::tianshu::launch_kernel<ReduceMaxOp, float>(
            _handle,
            ReduceMaxOp(),
            output,
            input,
            input_shape,
            output_shape,
            input_strides,
            output_strides,
            _axes,
            _keep_dims,
            workspace,
            workspace_size,
            stream);
    case INFINI_DTYPE_BF16:
        return reduce::tianshu::launch_kernel<ReduceMaxOp, bf16_t>(
            _handle,
            ReduceMaxOp(),
            output,
            input,
            input_shape,
            output_shape,
            input_strides,
            output_strides,
            _axes,
            _keep_dims,
            workspace,
            workspace_size,
            stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

size_t Descriptor::workspaceSize() const {
    return reduce::tianshu::get_workspace_size<ReduceMaxOp>(
        _input_desc->shape(),
        _output_desc->shape(),
        _dtype,
        _axes);
}

} // namespace op::reduce_max::tianshu