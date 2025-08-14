#include "reduceMean.h"
#include <ts_gpu_api.h> // 天数SDK头文件

infiniStatus_t reduceMean_tianshu(
    infiniTensorDescriptor_t input,
    infiniTensorDescriptor_t output,
    size_t dim
) {
    // 创建张量描述符
    tsTensorDescriptor input_desc, output_desc;
    tsStatus ts_status;
    
    // 初始化输入张量描述符
    ts_status = tsInitTensorDescriptor(&input_desc);
    if (ts_status != TS_SUCCESS) return INFINI_ERROR_DEVICE;
    
    // 设置输入张量描述
    tsDataType_t data_type;
    switch(input->dtype) {
        case INFINI_DTYPE_F32: data_type = TS_FLOAT; break;
        case INFINI_DTYPE_F16: data_type = TS_HALF; break;
        case INFINI_DTYPE_BF16: data_type = TS_BFLOAT16; break;
        default:
            tsDestroyTensorDescriptor(input_desc);
            return INFINI_ERROR_UNSUPPORTED_DTYPE;
    }
    
    ts_status = tsSetTensorNdDescriptor(
        input_desc,
        data_type,                 // 数据类型
        input->shape.size(),       // 维度数
        input->shape.data()        // 形状数组
    );
    if (ts_status != TS_SUCCESS) {
        tsDestroyTensorDescriptor(input_desc);
        return INFINI_ERROR_DEVICE;
    }
    
    // 初始化输出张量描述符
    ts_status = tsInitTensorDescriptor(&output_desc);
    if (ts_status != TS_SUCCESS) {
        tsDestroyTensorDescriptor(input_desc);
        return INFINI_ERROR_DEVICE;
    }
    
    // 设置输出形状 (保留维度，dim=1)
    std::vector<int> output_shape(input->shape.begin(), input->shape.end());
    output_shape[dim] = 1;
    
    ts_status = tsSetTensorNdDescriptor(
        output_desc,
        data_type,                 // 数据类型
        output_shape.size(),       // 维度数
        output_shape.data()        // 形状数组
    );
    if (ts_status != TS_SUCCESS) {
        tsDestroyTensorDescriptor(input_desc);
        tsDestroyTensorDescriptor(output_desc);
        return INFINI_ERROR_DEVICE;
    }
    
    // 执行规约操作
    ts_status = tsGpuReduce(
        input->data,              // 输入数据
        output->data,             // 输出数据
        input_desc,               // 输入描述
        output_desc,              // 输出描述
        TS_REDUCE_MEAN,           // 规约类型
        dim,                      // 规约维度
        0                         // 流ID
    );
    
    // 清理资源
    tsDestroyTensorDescriptor(input_desc);
    tsDestroyTensorDescriptor(output_desc);
    
    return (ts_status == TS_SUCCESS) ? INFINI_SUCCESS : INFINI_ERROR_DEVICE;
}