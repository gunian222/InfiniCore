#pragma once
#include "infiniCore.h"

INFINI_API infiniStatus_t reduceMean(
    infiniTensorDescriptor_t input,   // 输入张量
    infiniTensorDescriptor_t output,  // 输出张量
    size_t dim                        // 规约维度 (规范：snake_case)
);