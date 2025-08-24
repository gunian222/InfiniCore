#ifndef __REDUCE_MAX_INFO_H__
#define __REDUCE_MAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::reduce_max {

struct ReduceMaxInfo {
    infiniDtype_t dtype;
    int reduce_dim;
    size_t in_ndim;
    std::vector<size_t> in_shape;
    std::vector<size_t> in_stride;  
    std::vector<size_t> out_shape;
    std::vector<size_t> out_stride;  

    static utils::Result<ReduceMaxInfo> create(
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t output_desc,
        int reduce_dim) 
    {
        auto dtype = input_desc->dtype();

        // 输入 dtype 必须是数值型
        CHECK_DTYPE(dtype,
            INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
            INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        // 输出 dtype 要和输入一致
        CHECK_OR_RETURN(output_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

        size_t ndim = input_desc->ndim();
        CHECK_OR_RETURN(reduce_dim >= 0 && reduce_dim < (int)ndim, INFINI_STATUS_BAD_TENSOR_SHAPE);

        // 取输入 shape/stride
        std::vector<size_t> in_shape(ndim);
        std::vector<size_t> in_stride(ndim);
        for (size_t i = 0; i < ndim; i++) {
            in_shape[i]  = input_desc->dim(i);
            in_stride[i] = input_desc->stride(i);   // 从 desc 提取 stride
        }

        // 取输出 shape/stride
        std::vector<size_t> out_shape;
        std::vector<size_t> out_stride;
        for (size_t i = 0; i < output_desc->ndim(); i++) {
            out_shape.push_back(output_desc->dim(i));
            out_stride.push_back(output_desc->stride(i));
        }

        // 检查输出 shape 是否匹配
        std::vector<size_t> expect_out_shape;
        for (size_t i = 0; i < ndim; i++) {
            if (i != (size_t)reduce_dim) {
                expect_out_shape.push_back(in_shape[i]);
            }
        }

        CHECK_OR_RETURN(output_desc->ndim() == expect_out_shape.size(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        for (size_t i = 0; i < expect_out_shape.size(); i++) {
            CHECK_OR_RETURN(output_desc->dim(i) == expect_out_shape[i], INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        return utils::Result<ReduceMaxInfo>(
            {dtype, reduce_dim, ndim, in_shape, in_stride, out_shape, out_stride});
    }
};

} // namespace op::reduce_max

#endif // __REDUCE_MAX_INFO_H__
