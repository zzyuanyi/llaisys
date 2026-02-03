#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);

    // Check shapes: must have the same shape
    ASSERT(out->shape() == in->shape(), "Rearrange: output and input shapes must match");

    // Check dtypes
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // For now, only support CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(),
                             in->data(),
                             in->shape(),
                             out->strides(),
                             in->strides(),
                             out->dtype(),
                             out->elementSize());
    }

    // TODO: Add GPU support if needed
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
