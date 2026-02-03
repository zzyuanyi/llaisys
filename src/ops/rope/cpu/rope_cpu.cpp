#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {

    size_t half_dim = head_dim / 2;

    for (size_t seq = 0; seq < seq_len; ++seq) {
        size_t pos = static_cast<size_t>(pos_ids[seq]);

        for (size_t head = 0; head < n_heads; ++head) {
            for (size_t j = 0; j < half_dim; ++j) {
                // Calculate angle: φ_{seq,j} = pos * θ^(-2j/head_dim) = pos / θ^(2j/head_dim)
                float angle = static_cast<float>(pos) / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));

                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                // Get a and b values
                size_t a_idx = seq * n_heads * head_dim + head * head_dim + j;
                size_t b_idx = seq * n_heads * head_dim + head * head_dim + half_dim + j;

                float a_val, b_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a_val = llaisys::utils::cast<float>(in[a_idx]);
                    b_val = llaisys::utils::cast<float>(in[b_idx]);
                } else {
                    a_val = static_cast<float>(in[a_idx]);
                    b_val = static_cast<float>(in[b_idx]);
                }

                // Apply RoPE transformation
                float a_new = a_val * cos_val - b_val * sin_val;
                float b_new = b_val * cos_val + a_val * sin_val;

                // Store results
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[a_idx] = llaisys::utils::cast<T>(a_new);
                    out[b_idx] = llaisys::utils::cast<T>(b_new);
                } else {
                    out[a_idx] = static_cast<T>(a_new);
                    out[b_idx] = static_cast<T>(b_new);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                    reinterpret_cast<const float *>(in),
                    pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F64:
        return rope_(reinterpret_cast<double *>(out),
                    reinterpret_cast<const double *>(in),
                    pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                    reinterpret_cast<const llaisys::fp16_t *>(in),
                    pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                    reinterpret_cast<const llaisys::bf16_t *>(in),
                    pos_ids, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
