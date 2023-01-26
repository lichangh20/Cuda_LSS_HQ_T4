#include <torch/extension.h>
#include <tuple>

std::pair<torch::Tensor, std::vector<double>> quantize_cuda(torch::Tensor x, torch::Tensor qy, float scaley, torch::Tensor y, int num_bins);

std::pair<torch::Tensor, std::vector<double>> quantize(torch::Tensor x, torch::Tensor qy, float scaley, torch::Tensor y, int num_bins){
    TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor!");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous!");
    TORCH_CHECK(x.dim() == 2, "x must be 2D!");

    TORCH_CHECK(y.type().is_cuda(), "y must be a CUDA tensor!");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous!");
    TORCH_CHECK(y.dim() == 2, "y must be 2D!");

    return quantize_cuda(x, qy, scaley, y, num_bins);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
}