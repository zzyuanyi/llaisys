#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t total_elements) {
    for (size_t i = 0; i < total_elements; ++i) {
        float gate_val, up_val;

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            gate_val = llaisys::utils::cast<float>(gate[i]);
            up_val = llaisys::utils::cast<float>(up[i]);
        } else {
            gate_val = static_cast<float>(gate[i]);
            up_val = static_cast<float>(up[i]);
        }

        // Compute sigmoid: 1 / (1 + exp(-x))
        float sigmoid_val = 1.0f / (1.0f + std::exp(-gate_val));

        // Compute SwiGLU: up * gate * sigmoid(gate)
        float result = up_val * gate_val * sigmoid_val;

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(result);
        } else {
            out[i] = static_cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t total_elements) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                      reinterpret_cast<const float *>(gate),
                      reinterpret_cast<const float *>(up),
                      total_elements);
    case LLAISYS_DTYPE_F64:
        return swiglu_(reinterpret_cast<double *>(out),
                      reinterpret_cast<const double *>(gate),
                      reinterpret_cast<const double *>(up),
                      total_elements);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                      reinterpret_cast<const llaisys::fp16_t *>(gate),
                      reinterpret_cast<const llaisys::fp16_t *>(up),
                      total_elements);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                      reinterpret_cast<const llaisys::bf16_t *>(gate),
                      reinterpret_cast<const llaisys::bf16_t *>(up),
                      total_elements);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
