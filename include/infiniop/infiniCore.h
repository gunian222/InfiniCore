#pragma once
#include <vector>
#include <cstddef>

// 数据类型枚举
typedef enum {
    INFINI_DTYPE_F32 = 0,  // float32
    INFINI_DTYPE_F16,      // float16
    INFINI_DTYPE_BF16,     // bfloat16
    INFINI_DTYPE_INVALID
} infiniDtype_t;

// 设备类型枚举
typedef enum {
    INFINI_DEVICE_CUDA,    // 英伟达
    INFINI_DEVICE_MXDC,    // 沐曦
    INFINI_DEVICE_TIANSHU  // 天数
} infiniDeviceType_t;

// 错误码枚举
typedef enum {
    INFINI_SUCCESS = 0,
    INFINI_ERROR_INVALID_DIM,
    INFINI_ERROR_UNSUPPORTED_DTYPE,
    INFINI_ERROR_DEVICE_MISMATCH,
    INFINI_ERROR_PLATFORM_NOT_SUPPORTED
} infiniStatus_t;

// 张量描述符结构体
struct InfiniTensorDescriptor {
    void* data;                      // 数据指针
    std::vector<size_t> shape;       // 形状数组
    std::vector<size_t> stride;      // 步长数组
    infiniDtype_t dtype;             // 数据类型
    infiniDeviceType_t deviceType;   // 设备类型
    
    // 计算元素总数
    size_t getElementCount() const {
        size_t count = 1;
        for (auto s : shape) count *= s;
        return count;
    }
};

// 对外暴露的指针类型
typedef struct InfiniTensorDescriptor* infiniTensorDescriptor_t;

// API宏定义
#define INFINI_API