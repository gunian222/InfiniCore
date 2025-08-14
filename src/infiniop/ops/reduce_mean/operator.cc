#include "reduceMean.h"
#include "devices/reduceMean_cuda.h"
#include "devices/reduceMean_mxdc.h"
#include "devices/reduceMean_tianshu.h"

infiniStatus_t reduceMean(
    infiniTensorDescriptor_t input,
    infiniTensorDescriptor_t output,
    size_t dim
) {
    // 校验输入有效性
    if (dim >= input->shape.size()) {
        return INFINI_ERROR_INVALID_DIM;
    }
    
    // 校验输出形状
    if (input->shape.size() != output->shape.size()) {
        return INFINI_ERROR_DEVICE_MISMATCH;
    }
    for (size_t i = 0; i < input->shape.size(); i++) {
        if (i != dim && input->shape[i] != output->shape[i]) {
            return INFINI_ERROR_DEVICE_MISMATCH;
        }
    }
    if (output->shape[dim] != 1) {
        return INFINI_ERROR_DEVICE_MISMATCH;
    }
    
    // 设备路由
    switch(input->deviceType) {
        case INFINI_DEVICE_CUDA:
            return reduceMean_cuda(input, output, dim);
        case INFINI_DEVICE_MXDC:
            return reduceMean_mxdc(input, output, dim);
        case INFINI_DEVICE_TIANSHU:
            return reduceMean_tianshu(input, output, dim);
        default:
            return INFINI_ERROR_PLATFORM_NOT_SUPPORTED;
    }
}