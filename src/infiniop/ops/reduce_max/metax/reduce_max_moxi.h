#ifndef __REDUCE_MAX_MOXI_H__
#define __REDUCE_MAX_MOXI_H__

#include "reduce_max.h"


REDUCE_DESCRIPTOR(reduce_max, moxi)

namespace op::reduce_max::moxi {
typedef struct ReduceMaxOp {
public:
    static constexpr int num_inputs = 1;
    // 最大值规约操作
    template <typename T>
    inline __mxi_device__ T operator()(const T &a, const T &b) const {
        return max(a, b);
    }
    // 沐曦平台半精度优化
    inline __mxi_device__ mxi_half operator()(const mxi_half &a, const mxi_half &b) const {
        return mxi_hmax(a, b);
    }
} ReduceMaxOp;
} // namespace op::reduce_max::moxi

#endif // __REDUCE_MAX_MOXI_H__