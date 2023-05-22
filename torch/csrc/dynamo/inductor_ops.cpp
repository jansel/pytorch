#include <ATen/ops/from_blob.h>
#include <torch/library.h>

namespace {

static at::Tensor _tensor_from_blob(
    int64_t data_ptr,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    at::ScalarType dtype,
    at::Device device) {
  return at::for_blob(reinterpret_cast<void*>(data_ptr), size)
      .strides(stride)
      .options(at::TensorOptions().dtype(dtype).device(device))
      .target_device(device)
      .make_tensor();
}

TORCH_LIBRARY_FRAGMENT(inductor, m) {
  m.def(
      "_tensor_from_blob(int data_ptr, int[] size, int[] stride, ScalarType dtype, Device device) -> Tensor",
      &_tensor_from_blob);
}

} // namespace