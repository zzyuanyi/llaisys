#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    // Check shapes
    ASSERT(q->ndim() == 3, "Self-attention: query tensor must be 3D [qlen, nh, hd]");
    ASSERT(k->ndim() == 3, "Self-attention: key tensor must be 3D [kvlen, nkvh, hd]");
    ASSERT(v->ndim() == 3, "Self-attention: value tensor must be 3D [kvlen, nkvh, hd]");
    ASSERT(attn_val->ndim() == 3, "Self-attention: output tensor must be 3D [qlen, nh, hd]");

    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t hd = q->shape()[2];

    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    size_t k_hd = k->shape()[2];

    size_t v_kvlen = v->shape()[0];
    size_t v_nkvh = v->shape()[1];
    size_t v_hd = v->shape()[2];

    // Check output shape
    ASSERT(attn_val->shape()[0] == qlen, "Self-attention: output qlen must match query");
    ASSERT(attn_val->shape()[1] == nh, "Self-attention: output nh must match query");
    ASSERT(attn_val->shape()[2] == hd, "Self-attention: output hd must match query");

    // Check key/value shapes
    ASSERT(kvlen == v_kvlen, "Self-attention: key and value must have same kvlen");
    ASSERT(nkvh == v_nkvh, "Self-attention: key and value must have same nkvh");
    ASSERT(k_hd == hd && v_hd == hd, "Self-attention: key/value hd must match query hd");

    // Check head compatibility: nh must be divisible by nkvh
    ASSERT(nh % nkvh == 0, "Self-attention: nh must be divisible by nkvh for grouped query attention");

    // Check dtypes
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), k->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), v->dtype());

    // All tensors must be contiguous
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self-attention: all tensors must be contiguous");

    // For now, only support CPU
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(),
                                  q->data(),
                                  k->data(),
                                  v->data(),
                                  attn_val->dtype(),
                                  qlen, kvlen, nh, nkvh, hd, scale);
    }

    // TODO: Add GPU support if needed
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
