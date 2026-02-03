#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {

    // 输入形状：
    // q: [qlen, nh, hd]
    // k: [kvlen, nkvh, hd]
    // v: [kvlen, nkvh, hd]
    // attn_val: [qlen, nh, hd]

    size_t heads_per_kv = nh / nkvh;

    // 为注意力计算分配临时空间
    std::vector<float> attn_weights(qlen * nh * kvlen, 0.0f);
    std::vector<float> attn_probs(qlen * nh * kvlen, 0.0f);

    // Step 1: 计算注意力权重 Q @ K^T * scale
    for (size_t q_pos = 0; q_pos < qlen; ++q_pos) {
        for (size_t h = 0; h < nh; ++h) {
            for (size_t kv_pos = 0; kv_pos < kvlen; ++kv_pos) {
                float sum = 0.0f;

                // Q[q_pos, h, :] @ K[kv_pos, kv_h, :] where kv_h = h / (nh / nkvh)
                size_t heads_per_kv = nh / nkvh;
                size_t kv_h = h / heads_per_kv;

                for (size_t d = 0; d < hd; ++d) {
                    float q_val, k_val;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_pos * nh * hd + h * hd + d]);
                        k_val = llaisys::utils::cast<float>(k[kv_pos * nkvh * hd + kv_h * hd + d]);
                    } else {
                        q_val = static_cast<float>(q[q_pos * nh * hd + h * hd + d]);
                        k_val = static_cast<float>(k[kv_pos * nkvh * hd + kv_h * hd + d]);
                    }

                    sum += q_val * k_val;
                }

                // 应用缩放
                sum *= scale;

                // Apply causal mask: PyTorch style tril(diagonal=S-L)
                // diagonal = kvlen - qlen
                size_t diagonal = kvlen - qlen;
                if (kv_pos > q_pos + diagonal) {
                    sum = -std::numeric_limits<float>::infinity();
                }

                attn_weights[q_pos * nh * kvlen + h * kvlen + kv_pos] = sum;
            }
        }
    }

    // Step 2: 应用softmax按行
    for (size_t q_pos = 0; q_pos < qlen; ++q_pos) {
        for (size_t h = 0; h < nh; ++h) {
            // 找到最大值用于数值稳定性
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t kv_pos = 0; kv_pos < kvlen; ++kv_pos) {
                max_val = std::max(max_val, attn_weights[q_pos * nh * kvlen + h * kvlen + kv_pos]);
            }

            // 计算exp和总和
            float sum_exp = 0.0f;
            for (size_t kv_pos = 0; kv_pos < kvlen; ++kv_pos) {
                size_t idx = q_pos * nh * kvlen + h * kvlen + kv_pos;
                float exp_val = std::exp(attn_weights[idx] - max_val);
                attn_probs[idx] = exp_val;
                sum_exp += exp_val;
            }

            // 归一化
            for (size_t kv_pos = 0; kv_pos < kvlen; ++kv_pos) {
                size_t idx = q_pos * nh * kvlen + h * kvlen + kv_pos;
                attn_probs[idx] /= sum_exp;
            }
        }
    }

    // Step 3: 计算最终结果 attn_probs @ V
    for (size_t q_pos = 0; q_pos < qlen; ++q_pos) {
        for (size_t h = 0; h < nh; ++h) {
            for (size_t d = 0; d < hd; ++d) {
                float sum = 0.0f;

                for (size_t kv_pos = 0; kv_pos < kvlen; ++kv_pos) {
                    size_t kv_h = h / heads_per_kv;
                    float attn_prob = attn_probs[q_pos * nh * kvlen + h * kvlen + kv_pos];
                    float v_val;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[kv_pos * nkvh * hd + kv_h * hd + d]);
                    } else {
                        v_val = static_cast<float>(v[kv_pos * nkvh * hd + kv_h * hd + d]);
                    }

                    sum += attn_prob * v_val;
                }

                // 存储结果
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[q_pos * nh * hd + h * hd + d] = llaisys::utils::cast<T>(sum);
                } else {
                    attn_val[q_pos * nh * hd + h * hd + d] = static_cast<T>(sum);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F64:
        return self_attention_(reinterpret_cast<double *>(attn_val),
                              reinterpret_cast<const double *>(q),
                              reinterpret_cast<const double *>(k),
                              reinterpret_cast<const double *>(v),
                              qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              qlen, kvlen, nh, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
