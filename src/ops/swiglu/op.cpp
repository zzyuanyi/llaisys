#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);

    // Check shapes: all tensors should have the same shape
    ASSERT(out->shape() == gate->shape(), "SwiGLU: output and gate shapes must match");
    ASSERT(out->shape() == up->shape(), "SwiGLU: output and up shapes must match");

    // Check dtypes
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype());
    CHECK_SAME_DTYPE(out->dtype(), up->dtype());

    // All tensors must be contiguous
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous");

    // For now, only support CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(),
                          gate->data(),
                          up->data(),
                          out->dtype(),
                          out->numel());
    }

    // TODO: Add GPU support if needed
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
