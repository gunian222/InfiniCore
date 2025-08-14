#include "reduceMean.h"
#include <mxdc_api.h> // 沐曦SDK头文件

infiniStatus_t reduceMean_mxdc(
    infiniTensorDescriptor_t input,
    infiniTensorDescriptor_t output,
    size_t dim
) {
    // 创建规约描述符
    mxdcReduceDescriptor_t desc;
    mxdcStatus_t mx_status = mxdcCreateReduceDescriptor(&desc);
    if (mx_status != MXDC_SUCCESS) return INFINI_ERROR_DEVICE;
    
    // 设置规约参数
    mxdcSetReduceDescriptor(
        desc,
        MXDC_REDUCE_MEAN,        // 规约类型：求平均
        dim,                      // 规约维度
        input->shape.data(),      // 输入形状
        input->shape.size()       // 维度数
    );
    
    // 设置数据类型
    mxdcDataType_t data_type;
    switch(input->dtype) {
        case INFINI_DTYPE_F32: data_type = MXDC_FLOAT; break;
        case INFINI_DTYPE_F16: data_type = MXDC_HALF; break;
        case INFINI_DTYPE_BF16: data_type = MXDC_BFLOAT16; break;
        default:
            mxdcDestroyReduceDescriptor(desc);
            return INFINI_ERROR_UNSUPPORTED_DTYPE;
    }
    
    // 执行规约操作
    mx_status = mxdcReduce(
        desc,
        input->data,              // 输入数据
        output->data,             // 输出数据
        data_type,                // 数据类型
        0                         // 流ID
    );
    
    // 清理资源
    mxdcDestroyReduceDescriptor(desc);
    
    return (mx_status == MXDC_SUCCESS) ? INFINI_SUCCESS : INFINI_ERROR_DEVICE;
}