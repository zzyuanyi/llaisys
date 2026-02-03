#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,
               size_t batch_size, size_t feature_size, float eps) {

    for (size_t b = 0; b < batch_size; ++b) {
        // Calculate RMS for this row: sqrt( (1/d) * Σ(x_j²) + ε )
        float sum_squares = 0.0f;

        for (size_t f = 0; f < feature_size; ++f) {
            float x_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                x_val = llaisys::utils::cast<float>(in[b * feature_size + f]);
            } else {
                x_val = static_cast<float>(in[b * feature_size + f]);
            }
            sum_squares += x_val * x_val;
        }

        float rms = std::sqrt((sum_squares / static_cast<float>(feature_size)) + eps);

        // Apply normalization: y_i = w_i * x_i / rms
        for (size_t f = 0; f < feature_size; ++f) {
            float x_val, w_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                x_val = llaisys::utils::cast<float>(in[b * feature_size + f]);
                w_val = llaisys::utils::cast<float>(weight[f]);
            } else {
                x_val = static_cast<float>(in[b * feature_size + f]);
                w_val = static_cast<float>(weight[f]);
            }

            float result = w_val * x_val / rms;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[b * feature_size + f] = llaisys::utils::cast<T>(result);
            } else {
                out[b * feature_size + f] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t batch_size, size_t feature_size, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                        reinterpret_cast<const float *>(in),
                        reinterpret_cast<const float *>(weight),
                        batch_size, feature_size, eps);
    case LLAISYS_DTYPE_F64:
        return rms_norm_(reinterpret_cast<double *>(out),
                        reinterpret_cast<const double *>(in),
                        reinterpret_cast<const double *>(weight),
                        batch_size, feature_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight),
                        batch_size, feature_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight),
                        batch_size, feature_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
