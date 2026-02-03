#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight,
                size_t num_indices, size_t embedding_dim, size_t vocab_size) {
    for (size_t i = 0; i < num_indices; ++i) {
        size_t idx = static_cast<size_t>(index[i]);
        if (idx >= vocab_size) {
            // Handle out of bounds - this should not happen in normal usage
            // but we can set to zeros or handle gracefully
            memset(out + i * embedding_dim, 0, embedding_dim * sizeof(T));
            continue;
        }

        // Copy the embedding vector for this index
        const T *src = weight + idx * embedding_dim;
        T *dst = out + i * embedding_dim;
        memcpy(dst, src, embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t num_indices, size_t embedding_dim, size_t vocab_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), index,
                         reinterpret_cast<const float *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_F64:
        return embedding_(reinterpret_cast<double *>(out), index,
                         reinterpret_cast<const double *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), index,
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), index,
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_I8:
        return embedding_(reinterpret_cast<int8_t *>(out), index,
                         reinterpret_cast<const int8_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_I16:
        return embedding_(reinterpret_cast<int16_t *>(out), index,
                         reinterpret_cast<const int16_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_I32:
        return embedding_(reinterpret_cast<int32_t *>(out), index,
                         reinterpret_cast<const int32_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_I64:
        return embedding_(reinterpret_cast<int64_t *>(out), index,
                         reinterpret_cast<const int64_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_U8:
        return embedding_(reinterpret_cast<uint8_t *>(out), index,
                         reinterpret_cast<const uint8_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_U16:
        return embedding_(reinterpret_cast<uint16_t *>(out), index,
                         reinterpret_cast<const uint16_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_U32:
        return embedding_(reinterpret_cast<uint32_t *>(out), index,
                         reinterpret_cast<const uint32_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_U64:
        return embedding_(reinterpret_cast<uint64_t *>(out), index,
                         reinterpret_cast<const uint64_t *>(weight),
                         num_indices, embedding_dim, vocab_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
