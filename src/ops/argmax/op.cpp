#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // Check shapes: vals should be 1D, max_idx and max_val should be 1D with size 1
    ASSERT(vals->ndim() == 1, "Argmax: input tensor must be 1D");
    ASSERT(max_idx->ndim() == 1 && max_idx->shape()[0] == 1, "Argmax: max_idx must be 1D with size 1");
    ASSERT(max_val->ndim() == 1 && max_val->shape()[0] == 1, "Argmax: max_val must be 1D with size 1");

    // Check dtypes: max_idx should be I64, max_val should match vals dtype
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be I64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    // All tensors must be contiguous
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous");

    // For now, only support CPU
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    // TODO: Add GPU support if needed
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
