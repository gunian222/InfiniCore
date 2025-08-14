#ifndef __INFINIOP_REDUCEMEAN_API_H__
#define __INFINIOP_REDUCEMEAN_API_H__

#include "../infinicore.h"


typedef struct InfiniopDescriptor *infiniopReduceMeanDescriptor_t;

__C __export infiniStatus_t infiniopCreateReduceMeanDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMeanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    const int *axes,
    int num_axes,
    int keep_dims);


__C __export infiniStatus_t infiniopGetReduceMeanWorkspaceSize(
    infiniopReduceMeanDescriptor_t desc,
    size_t *size);


__C __export infiniStatus_t infiniopReduceMean(
    infiniopReduceMeanDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t infiniopDestroyReduceMeanDescriptor(
    infiniopReduceMeanDescriptor_t desc);

#endif // __INFINIOP_REDUCEMEAN_API_H__
