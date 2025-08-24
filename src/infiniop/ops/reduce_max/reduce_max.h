#ifndef __REDUCE_MAX_H__
#define __REDUCE_MAX_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "info.h"

#define REDUCE_DESCRIPTOR(NAMESPACE)                               \
                                                                    \
    namespace op::reduce_max::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {           \
        struct Opaque;                                              \
        Opaque *_opaque;                                            \
        ReduceMaxInfo _info;                                        \
                                                                     \
        Descriptor(                                                 \
            ReduceMaxInfo info,                                     \
            Opaque *opaque,                                         \
            infiniDevice_t device_type,                             \
            int device_id)                                          \
            : InfiniopDescriptor{device_type, device_id},           \
              _opaque(opaque),                                      \
              _info(std::move(info)) {}                             \
                                                                     \
    public:                                                         \
        ~Descriptor();                                              \
                                                                     \
        static infiniStatus_t create(                               \
            infiniopHandle_t handle,                                \
            Descriptor **desc_ptr,                                  \
            infiniopTensorDescriptor_t y_desc,                      \
            infiniopTensorDescriptor_t x_desc,                      \
            const int* axes,                                        \
            int num_axes,                                           \
            int keep_dims)                                          \
        {                                                           \
                                                                   \
            if (num_axes != 1) return INFINI_STATUS_NOT_SUPPORTED;  \
                                                                     \
            auto maybe_info = ReduceMaxInfo::create(x_desc, y_desc, axes[0]); \
            if (!maybe_info.ok())                                   \
                return maybe_info.status();                         \
                                                                     \
            auto info = maybe_info.value();                         \
                                                                     \
                                                                    \
            Opaque *opaque = nullptr;                               \
                                                                     \
            *desc_ptr = new Descriptor(info, opaque,                \
                handle->device_type, handle->device_id);            \
            return INFINI_STATUS_SUCCESS;                           \
        }                                                           \
                                                                     \
        infiniStatus_t calculate(                                   \
            void *y,                                                \
            const void *x,                                          \
            void *stream) const;                                    \
    };                                                              \
    }

#endif // __REDUCE_MAX_H__
