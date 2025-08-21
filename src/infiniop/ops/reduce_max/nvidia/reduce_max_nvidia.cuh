#ifndef __REDUCE_MAX_NVIDIA_CUH__
#define __REDUCE_MAX_NVIDIA_CUH__

#include "../../../reduce/nvidia/reduce_nvidia_api.cuh"

REDUCE_DESCRIPTOR(reduce_max, nvidia)

namespace op::reduce_max::nvidia {
typedef struct ReduceMaxOp {
public:
    static constexpr int num_inputs = 1;
    // 最大值规约操作
    template <typename T>
    inline __device__ T operator()(const T &a, const T &b) const {
        return max(a, b);
    }
    // 半精度优化实现
    inline __device__ half operator()(const half &a, const half &b) const {
        return __hmax(a, b);
    }
    // bfloat16特化实现
    inline __device__ __nv_bfloat16 operator()(const __nv_bfloat16 &a, const __nv_bfloat16 &b) const {
        return (__half2float(a) > __half2float(b)) ? a : b;
    }
} ReduceMaxOp;
} // namespace op::reduce_max::nvidia

#endif // __REDUCE_MAX_NVIDIA_CUH__