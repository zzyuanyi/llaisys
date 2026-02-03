#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    // Check shapes
    ASSERT(in->ndim() == 2, "Linear: input tensor must be 2D");
    ASSERT(weight->ndim() == 2, "Linear: weight tensor must be 2D");
    ASSERT(out->ndim() == 2, "Linear: output tensor must be 2D");

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];  // weight is (out_features, in_features)
    size_t weight_in_features = weight->shape()[1];

    ASSERT(out->shape()[0] == batch_size, "Linear: output batch size must match input");
    ASSERT(out->shape()[1] == out_features, "Linear: output features must match weight out features");
    ASSERT(weight_in_features == in_features, "Linear: weight in features must match input features");

    if (bias) {
        ASSERT(bias->ndim() == 1, "Linear: bias tensor must be 1D");
        ASSERT(bias->shape()[0] == out_features, "Linear: bias size must match output features");
    }

    // Check dtypes
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    // All tensors must be contiguous
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: input, weight, and output tensors must be contiguous");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias tensor must be contiguous");
    }

    // For now, only support CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        bool has_bias = (bias != nullptr);
        return cpu::linear(out->data(),
                          in->data(),
                          weight->data(),
                          has_bias ? bias->data() : nullptr,
                          out->dtype(),
                          batch_size,
                          in_features,
                          out_features,
                          has_bias);
    }

    // TODO: Add GPU support if needed
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
