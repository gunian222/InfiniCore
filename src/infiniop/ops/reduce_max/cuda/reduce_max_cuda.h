#ifndef __REDUCE_MAX_CUDA_H__
#define __REDUCE_MAX_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace op::reduce_max::cuda {

typedef struct ReduceMaxOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (a > b) ? a : b;
    }

    // 半精度特化
    __device__ __forceinline__ half operator()(const half &a, const half &b) const {
        return __hgt(a, b) ? a : b;
    }

    // 半精度向量 (half2)
    __device__ __forceinline__ half2 operator()(const half2 &a, const half2 &b) const {
        return __hmax2(a, b);
    }

    // bfloat16 特化
    __device__ __forceinline__ __nv_bfloat16 operator()(const __nv_bfloat16 &a,
                                                        const __nv_bfloat16 &b) const {
        return (__bfloat162float(a) > __bfloat162float(b)) ? a : b;
    }
} ReduceMaxOp;

} // namespace op::reduce_max::cuda

#endif // __REDUCE_MAX_CUDA_H__
