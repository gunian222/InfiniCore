#ifndef __REDUCE_MAX_TIANSHU_H__
#define __REDUCE_MAX_TIANSHU_H__
#include "reduce_max.h"

REDUCE_DESCRIPTOR(reduce_max, tianshu)

namespace op::reduce_max::tianshu {
typedef struct ReduceMaxOp {
public:
    static constexpr int num_inputs = 1;
    // 最大值规约操作
    template <typename T>
    inline __device__ T operator()(const T &a, const T &b) const {
        return (a > b) ? a : b;
    }
    // 天数平台bfloat16处理
    inline __device__ bfloat16_t operator()(const bfloat16_t &a, const bfloat16_t &b) const {
        float a_f = bfloat16_to_float(a);
        float b_f = bfloat16_to_float(b);
        return float_to_bfloat16((a_f > b_f) ? a_f : b_f);
    }
} ReduceMaxOp;
} // namespace op::reduce_max::tianshu

#endif // __REDUCE_MAX_TIANSHU_H__