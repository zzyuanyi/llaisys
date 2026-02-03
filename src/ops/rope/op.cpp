#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // Check shapes
    ASSERT(in->ndim() == 3, "RoPE: input tensor must be 3D [seq_len, n_heads, head_dim]");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids tensor must be 1D [seq_len]");
    ASSERT(out->ndim() == 3, "RoPE: output tensor must be 3D [seq_len, n_heads, head_dim]");

    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    ASSERT(out->shape()[0] == seq_len, "RoPE: output seq_len must match input");
    ASSERT(out->shape()[1] == n_heads, "RoPE: output n_heads must match input");
    ASSERT(out->shape()[2] == head_dim, "RoPE: output head_dim must match input");
    ASSERT(pos_ids->shape()[0] == seq_len, "RoPE: pos_ids length must match seq_len");

    // Head dimension must be even for RoPE
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even");

    // Check dtypes
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be I64");

    // All tensors must be contiguous
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous");

    // For now, only support CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(),
                        in->data(),
                        reinterpret_cast<const int64_t *>(pos_ids->data()),
                        out->dtype(),
                        seq_len,
                        n_heads,
                        head_dim,
                        theta);
    }

    // TODO: Add GPU support if needed
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
