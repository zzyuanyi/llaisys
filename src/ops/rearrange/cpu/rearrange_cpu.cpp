#include "rearrange_cpu.hpp"

#include <cstring>
#include <vector>

namespace llaisys::ops::cpu {

// Helper function to recursively copy data
void rearrange_recursive(std::byte *out, const std::byte *in,
                        const std::vector<size_t> &shape,
                        const std::vector<ptrdiff_t> &out_strides,
                        const std::vector<ptrdiff_t> &in_strides,
                        size_t element_size, size_t dim, size_t out_offset, size_t in_offset) {
    if (dim == shape.size()) {
        // Copy single element
        std::memcpy(out + out_offset, in + in_offset, element_size);
        return;
    }

    for (size_t i = 0; i < shape[dim]; ++i) {
        rearrange_recursive(out, in, shape, out_strides, in_strides, element_size,
                           dim + 1,
                           out_offset + i * out_strides[dim] * element_size,
                           in_offset + i * in_strides[dim] * element_size);
    }
}

void rearrange(std::byte *out, const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t dtype, size_t element_size) {
    rearrange_recursive(out, in, shape, out_strides, in_strides, element_size, 0, 0, 0);
}
} // namespace llaisys::ops::cpu
