#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t num_indices, size_t embedding_dim, size_t vocab_size);
}
