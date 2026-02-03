#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    // Check shapes
    ASSERT(in->ndim() == 2, "RMS Norm: input tensor must be 2D");
    ASSERT(weight->ndim() == 1, "RMS Norm: weight tensor must be 1D");
    ASSERT(out->ndim() == 2, "RMS Norm: output tensor must be 2D");

    size_t batch_size = in->shape()[0];
    size_t feature_size = in->shape()[1];

    ASSERT(out->shape()[0] == batch_size, "RMS Norm: output batch size must match input");
    ASSERT(out->shape()[1] == feature_size, "RMS Norm: output feature size must match input");
    ASSERT(weight->shape()[0] == feature_size, "RMS Norm: weight size must match input feature size");

    // Check dtypes
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // All tensors must be contiguous
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMS Norm: all tensors must be contiguous");

    // For now, only support CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(),
                           in->data(),
                           weight->data(),
                           out->dtype(),
                           batch_size,
                           feature_size,
                           eps);
    }

    // TODO: Add GPU support if needed
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
