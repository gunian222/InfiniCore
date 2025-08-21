#ifndef __INFINIOP_REDUCEMAX_API_H__
#define __INFINIOP_REDUCEMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopReduceMaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateReduceMaxDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    const int *axes,
    int num_axes,
    int keep_dims);

__C __export infiniStatus_t infiniopGetReduceMaxWorkspaceSize(
    infiniopReduceMaxDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopReduceMax(
    infiniopReduceMaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t infiniopDestroyReduceMaxDescriptor(
    infiniopReduceMaxDescriptor_t desc);

#endif // __INFINIOP_REDUCEMAX_API_H__