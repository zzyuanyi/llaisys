#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        *max_idx = -1;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            *max_val = llaisys::utils::cast<T>(-std::numeric_limits<float>::max());
        } else {
            *max_val = std::numeric_limits<T>::lowest();
        }
        return;
    }

    size_t max_index = 0;
    T max_value = vals[0];

    for (size_t i = 1; i < numel; ++i) {
        bool is_greater = false;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            is_greater = llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(max_value);
        } else {
            is_greater = vals[i] > max_value;
        }

        if (is_greater) {
            max_value = vals[i];
            max_index = i;
        }
    }

    *max_idx = static_cast<int64_t>(max_index);
    *max_val = max_value;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                      reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_F64:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<double *>(max_val),
                      reinterpret_cast<const double *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                      reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                      reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_I8:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<int8_t *>(max_val),
                      reinterpret_cast<const int8_t *>(vals), numel);
    case LLAISYS_DTYPE_I16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<int16_t *>(max_val),
                      reinterpret_cast<const int16_t *>(vals), numel);
    case LLAISYS_DTYPE_I32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<int32_t *>(max_val),
                      reinterpret_cast<const int32_t *>(vals), numel);
    case LLAISYS_DTYPE_I64:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<int64_t *>(max_val),
                      reinterpret_cast<const int64_t *>(vals), numel);
    case LLAISYS_DTYPE_U8:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<uint8_t *>(max_val),
                      reinterpret_cast<const uint8_t *>(vals), numel);
    case LLAISYS_DTYPE_U16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<uint16_t *>(max_val),
                      reinterpret_cast<const uint16_t *>(vals), numel);
    case LLAISYS_DTYPE_U32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<uint32_t *>(max_val),
                      reinterpret_cast<const uint32_t *>(vals), numel);
    case LLAISYS_DTYPE_U64:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<uint64_t *>(max_val),
                      reinterpret_cast<const uint64_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
