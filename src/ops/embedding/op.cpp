#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    // Check shapes: index should be 1D, weight should be 2D, out should be (index.size(), weight.shape[1])
    ASSERT(index->ndim() == 1, "Embedding: index tensor must be 1D");
    ASSERT(weight->ndim() == 2, "Embedding: weight tensor must be 2D");
    ASSERT(out->ndim() == 2, "Embedding: output tensor must be 2D");
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: output first dim must match index size");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: output second dim must match weight second dim");

    // Check dtypes: index should be I64, out and weight should have same dtype
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be I64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // All tensors must be contiguous
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous");

    // For now, only support CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(),
                             reinterpret_cast<const int64_t *>(index->data()),
                             weight->data(),
                             weight->dtype(),
                             index->numel(),
                             weight->shape()[1],
                             weight->shape()[0]);
    }

    // TODO: Add GPU support if needed
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
