#ifndef __REDUCE_MAX_H__
#define __REDUCE_MAX_H__

#include "../../../utils.h"
#include "../../operator.h"

#define REDUCE_DESCRIPTOR(NAMESPACE)                               \
                                                                    \
    namespace op::reduce_max::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {           \
        struct Opaque;                                              \
        Opaque *_opaque;                                            \
        utils::ReduceMeta _meta; /* 存储 reduce 维度、输入输出信息 */ \
                                                                     \
        Descriptor(                                                 \
            utils::ReduceMeta meta,                                 \
            Opaque *opaque,                                         \
            infiniDevice_t device_type,                             \
            int device_id)                                          \
            : InfiniopDescriptor{device_type, device_id},           \
              _opaque(opaque),                                      \
              _meta(meta) {}                                        \
                                                                     \
    public:                                                         \
        ~Descriptor();                                              \
                                                                     \
        static infiniStatus_t create(                               \
            infiniopHandle_t handle,                                \
            Descriptor **desc_ptr,                                  \
            infiniopTensorDescriptor_t y_desc,                      \
            infiniopTensorDescriptor_t x_desc,                      \
            int reduce_dim); /* 新增参数：指定 reduce 的维度 */      \
                                                                     \
        infiniStatus_t calculate(                                   \
            void *y,                                                \
            const void *x,                                          \
            void *stream) const;                                    \
    };                                                              \
    }

#endif // __REDUCE_MAX_H__
