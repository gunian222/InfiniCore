#include <vector>
#include <cstddef>
#include <cmath>
#include <cstring>

// 数据类型枚举 (符合规范：infinixx[XxxXxx]_t)
typedef enum {
    INFINI_DTYPE_F32,
    INFINI_DTYPE_F16,
    INFINI_DTYPE_BF16,
    INFINI_DTYPE_INVALID
} infiniDtype_t;

// 张量描述符 (符合规范：内部类型 UpperCamelCase)
struct InfiniTensorDescriptor {
    void* data;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    infiniDtype_t dtype;
};

// 对外暴露指针类型 (符合规范：infinixx[XxxXxx]_t)
typedef struct InfiniTensorDescriptor *infiniTensorDescriptor_t;

// 辅助函数：半精度转换 (f16/bf16)
inline float halfToFloat(uint16_t value, bool isBf16) {
    if (isBf16) {
        // 简化版bf16转换
        uint32_t tmp = value << 16;
        return *reinterpret_cast<float*>(&tmp);
    } else {
        // 简化版f16转换
        uint32_t tmp = ((value & 0x8000) << 16) |
                      (((value & 0x7C00) + 0x1C000) << 13) |
                      ((value & 0x03FF) << 13);
        return *reinterpret_cast<float*>(&tmp);
    }
}

inline uint16_t floatToHalf(float value, bool isBf16) {
    uint32_t tmp = *reinterpret_cast<uint32_t*>(&value);
    if (isBf16) {
        return tmp >> 16;  // 取高16位
    } else {
        // 简化版f16转换
        return ((tmp >> 16) & 0x8000) |
               ((((tmp & 0x7F800000) - 0x38000000) >> 13) & 0x7C00) |
               ((tmp >> 13) & 0x03FF);
    }
}