#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <ctime>
#include "cuda_fp16.hpp"
#include "cuda_fp16.h"

#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <curand_kernel.h>
#include <tuple>
#include <bits/stdc++.h>

#include "torch/script.h"
using namespace torch::indexing;

template<typename scalar_t>
__global__ void first_quantize_cuda_kernel(const scalar_t * __restrict__  MatI, int8_t * MatO_transform, scalar_t * __restrict__  MatO_quantize, scalar_t * __restrict__  MatO_residual, const float scale, const float zero_point, int size){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        scalar_t input = MatI[x];
        scalar_t tmp1 = (input - zero_point) * scale - 8;
        int tmp2 = tmp1;
        int bias = (tmp1 - tmp2) * 2;
        int transform = std::clamp(tmp2+bias, -8, 7);
        MatO_transform[x] = transform;
        scalar_t quantize = (transform + 8) / scale + zero_point;
        MatO_quantize[x] = quantize;
        MatO_residual[x] = input - quantize;
    }
}

template<typename scalar_t>
__global__ void second_quantize_cuda_kernel(const scalar_t * __restrict__  MatI, int8_t * MatO_transform, scalar_t * __restrict__  MatO_quantize, const float scale, const float  zero_point, int size, unsigned long seed){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        // set random value
        curandStatePhilox4_32_10_t state;
        curand_init(seed, x, 0, &state);
        const float noise = curand_uniform(&state);

        // scalar_t tmp = (MatI[x] - zero_point[0]) * scale[0] + noise - 0.5 - 8;
        scalar_t tmp1 = (MatI[x] - zero_point) * scale - 8;
        int tmp2 = tmp1;
        int bias = (tmp1 - tmp2) * 2;
        MatO_transform[x] = std::clamp(tmp2+bias, -8, 7);
        MatO_quantize[x] = (MatO_transform[x] + 8) / scale + zero_point;
    }
}

__global__ void pack_cuda_kernel(int8_t * in, int8_t * out, int size){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        out[x] = (in[(x<<1)+1] << 4) | (in[x<<1] & 15);
    }
}

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  const int M,
  const int N,
  const int K,
  const cutlass::int4b_t *A,
  int lda,
  const cutlass::int4b_t *B,
  int ldb,
  int32_t *C,
  int ldc) {

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::int4b_t;                       // <- data type of elements in input matrix A
using ElementInputB = cutlass::int4b_t;                       // <- data type of elements in input matrix B
using ElementOutput = int32_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<256, 128, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 64 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 32>;  // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    4,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
    
  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A, lda},  // <- reference to matrix A on device
                                     {B, ldb},  // <- reference to matrix B on device
                                     {C, ldc},  // <- reference to matrix C on device
                                     {C, ldc},  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
  
    Gemm gemm_op;
    cutlass::Status status = gemm_op(arguments);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

#define N_THREADS 256

template<typename scalar_t>
__global__ void dequantize_cuda_kernel(const int32_t * gemm1, const int32_t * gemm2, scalar_t * __restrict__ output, 
                                        const int64_t * sum_y_column, const float const_x,
                                        const float scale_gemm1, const float scale_gemm2, int size, int ny){  
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int row = x / ny, col = x - row * ny;
    int sumY = sum_y_column[col];

    if (x<size){
       output[x] = gemm1[x] * scale_gemm1 + gemm2[x] * scale_gemm2 + const_x * sumY;
    }
}

__device__ __inline__ c10::Half __shfl_down_sync(const unsigned mask, const c10::Half var,
                                                 const unsigned int delta, const int width) {
  __half var_ = var;
  return __shfl_down_sync(mask, var_, delta, width);
}

//TODO: N means rows, D means cols
template<typename scalar_t>
__global__ void linalg_norm_cuda_kernel(const scalar_t * __restrict__ in, scalar_t * __restrict__ linalg, int N, int D, int stride_D){
  scalar_t sum_val;
  sum_val = 0;

  for (int64_t k1_outer = 0; k1_outer < stride_D; ++k1_outer) {
    scalar_t temp = in[blockIdx.x * D + (k1_outer << 5) + threadIdx.x];
    sum_val += temp * temp;
  }

  unsigned int mask;
  scalar_t sum_val_t;
  mask = __activemask();

  sum_val_t = __shfl_down_sync(mask, sum_val, 16, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 8, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 4, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 2, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 1, 32);
  sum_val += sum_val_t;
  linalg[blockIdx.x] = sqrt(sum_val);
}


std::pair<torch::Tensor, std::vector<double>> quantize_cuda(torch::Tensor x, torch::Tensor qy, float scaley, torch::Tensor y, int num_bins){
    std::vector<double> time_vector;
    int nz = x.size(0);
    int nx = x.size(1);
    int ny = y.size(1);

    cudaDeviceSynchronize();
    clock_t time_quantize1_start = clock();

    auto option_transform = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto option_quantize = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor first_transform = torch::empty({nz, nx}, option_transform);
    torch::Tensor first_quantize = torch::empty({nz, nx}, option_quantize);
    torch::Tensor first_residual = torch::empty({nz, nx}, option_quantize);

    dim3 block(N_THREADS);
    dim3 grid1((nx*nz-1)/block.x+1);
    int size_quantize = nz * nx ;
    // process of first quantize
    float mn = std::min(x.min().item<float>() - 1e-8, 0.);
    float mx = std::max(x.max().item<float>() + 1e-8, 0.);

    float zero_point1 = mn;
    float scale1 = num_bins / (mx - mn);

    float iqzero = floor(-zero_point1 * scale1);

    if (fabs(iqzero) < 1e-10){
        zero_point1 = 0;
        mn = 0;
    } else if (iqzero > 0){
        mx = (iqzero - num_bins) * mn / iqzero;
    }
    scale1 = num_bins / (mx - mn);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "first_quantize_cuda", ([&] {
    first_quantize_cuda_kernel<scalar_t><<<grid1, block>>>(
        x.data_ptr<scalar_t>(),
        first_transform.data_ptr<int8_t>(),
        first_quantize.data_ptr<scalar_t>(),
        first_residual.data_ptr<scalar_t>(),
        scale1, zero_point1,
        size_quantize);
    }));

    cudaDeviceSynchronize();
    clock_t time_quantize1_end = clock();
    // process of second quantize
    torch::Tensor second_transform = torch::empty({nz, nx}, option_transform);
    torch::Tensor second_quantize = torch::empty({nz, nx}, option_quantize);

    mn = std::min(first_residual.min().item<float>() - 1e-8, 0.);
    mx = std::max(first_residual.max().item<float>() + 1e-8, 0.);

    float zero_point2 = mn;
    float scale2 = num_bins / (mx - mn);

    iqzero = floor(-zero_point2 * scale2);

    if (fabs(iqzero) < 1e-10){
        zero_point2 = 0;
        mn = 0;
    } else if (iqzero > 0){
        mx = (iqzero - num_bins) * mn / iqzero;
    }
    scale2 = num_bins / (mx - mn);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "second_quantize_cuda", ([&] {
    second_quantize_cuda_kernel<scalar_t><<<grid1, block>>>(
        first_residual.data_ptr<scalar_t>(),
        second_transform.data_ptr<int8_t>(),
        second_quantize.data_ptr<scalar_t>(),
        scale2, zero_point2, 
        size_quantize,rand());
    }));

    cudaDeviceSynchronize();
    clock_t time_quantize2_end = clock();

    // leverage score
    // TODO: use dim=0 because torch.linalg only supports dim=1
    int threads = 32;
    int blocks = nz;

    auto x1_len = torch::empty({nz,}, option_quantize);
    auto x2_len = torch::empty({nz,}, option_quantize);
    auto y_len = torch::empty({nz,}, option_quantize);

    int stride_x = nx / 32;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1_len.scalar_type(), "linalg_cuda", ([&] {
    linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
        first_quantize.data_ptr<scalar_t>(), 
        x1_len.data_ptr<scalar_t>(),
        nz,nx,stride_x);
    }));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x2_len.scalar_type(), "linalg_cuda", ([&] {
    linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
        second_quantize.data_ptr<scalar_t>(), 
        x2_len.data_ptr<scalar_t>(),
        nz,nx,stride_x);
    }));
    int stride_y = ny / 32;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "linalg_cuda", ([&] {
    linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
        y.data_ptr<scalar_t>(), 
        y_len.data_ptr<scalar_t>(),
        nz,ny,stride_y);
    }));

    auto vec_norm = torch::cat({torch::mul(x1_len, y_len), torch::mul(x2_len, y_len)});

    cudaDeviceSynchronize();
    clock_t time_leverage_end = clock();

    //TODO: suppose an easy situation so that it can be faster
    int half_xshape = first_transform.size(0) / 2;
    auto sample_x1 = first_transform.index({Slice({None, half_xshape})}).t().contiguous();
    auto sample_x2 = second_transform.index({Slice({None, half_xshape})}).t().contiguous();
    int half_yshape = qy.size(0) / 2;
    auto sample_y = qy.index({Slice({None, half_yshape})}).t().contiguous();

    cudaDeviceSynchronize();
    clock_t time_sample_end = clock();

    // pack process
    auto sample_x1_int4 = torch::empty({nx, nz>>2}, option_transform);
    auto sample_x2_int4 = torch::empty({nx, nz>>2}, option_transform);
    auto sample_y_int4 = torch::empty({ny, nz>>2}, option_transform);
    int grid_size_x = nx*nz/4;
    int grid_size_y = nz*ny/4;
    dim3 grid_pack_x((grid_size_x-1)/block.x+1);
    dim3 grid_pack_y((grid_size_y-1)/block.x+1);
    pack_cuda_kernel<<<grid_pack_x,block>>>(sample_x1.data_ptr<int8_t>(), sample_x1_int4.data_ptr<int8_t>(), grid_size_x);
    pack_cuda_kernel<<<grid_pack_x,block>>>(sample_x2.data_ptr<int8_t>(), sample_x2_int4.data_ptr<int8_t>(), grid_size_x);
    pack_cuda_kernel<<<grid_pack_y,block>>>(sample_y.data_ptr<int8_t>(), sample_y_int4.data_ptr<int8_t>(), grid_size_y);


    cudaDeviceSynchronize();
    clock_t time_pack_end = clock();

    // gemm process
    cudaError_t result;
    int lda = nz / 2;
    int ldb = nz / 2;
    int ldc = ny;
    // Chunked matrix multiplication
    auto gemm1 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    result = CutlassSgemmNN(nx, ny, nz/2, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda, reinterpret_cast<cutlass::int4b_t *>(sample_y_int4.data_ptr<int8_t>()), ldb, gemm1.data_ptr<int32_t>(), ldc);

    cudaDeviceSynchronize();
    clock_t time_gemm1_end = clock();

    auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    result = CutlassSgemmNN(nx, ny, nz/2, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda, reinterpret_cast<cutlass::int4b_t *>(sample_y_int4.data_ptr<int8_t>()), ldb, gemm2.data_ptr<int32_t>(), ldc);
    cudaDeviceSynchronize();
    clock_t time_gemm2_end = clock();

    // dequantize process
    dim3 grid2((nx*ny-1)/block.x+1);
    // First dequantize higher 4 bits
    auto option_output = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto sum_y_column = torch::sum(sample_y, 1);
    auto output_final = torch::empty({nx,ny}, option_output);

    float const_x1 = (8.0 / scale1 + zero_point1) / scaley;
    float const_x2 = (8.0 / scale2 + zero_point2) / scaley;
    float const_x = const_x1 + const_x2;
    float scale_gemm1 = 1./ (scale1 * scaley);
    float scale_gemm2 = 1./ (scale2 * scaley);
    int size = nx*ny;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_final.scalar_type(), "dequantize_cuda", ([&] {
    dequantize_cuda_kernel<scalar_t><<<grid2, block>>>(
        gemm1.data_ptr<int32_t>(), 
        gemm2.data_ptr<int32_t>(),
        output_final.data_ptr<scalar_t>(),
        sum_y_column.data_ptr<int64_t>(),
        const_x, scale_gemm1, scale_gemm2,
        size, ny);
    }));

    cudaDeviceSynchronize();
    clock_t time_dequantize_end = clock();

    double quantize1_time = (double)(time_quantize1_end - time_quantize1_start) / CLOCKS_PER_SEC;
    double quantize2_time = (double)(time_quantize2_end - time_quantize1_end) / CLOCKS_PER_SEC;
    double leverage_time = (double)(time_leverage_end - time_quantize2_end) / CLOCKS_PER_SEC;
    double sample_time = (double)(time_sample_end - time_leverage_end) / CLOCKS_PER_SEC;
    double pack_time = (double)(time_pack_end - time_sample_end) / CLOCKS_PER_SEC;
    double gemm1_time = (double)(time_gemm1_end - time_pack_end) / CLOCKS_PER_SEC;
    double gemm2_time = (double)(time_gemm2_end - time_gemm1_end) / CLOCKS_PER_SEC;
    double dequantize_time = (double)(time_dequantize_end - time_gemm2_end) / CLOCKS_PER_SEC;
    // time_leverage_end

    time_vector.push_back(quantize1_time);
    time_vector.push_back(quantize2_time);
    time_vector.push_back(leverage_time);
    time_vector.push_back(sample_time);
    time_vector.push_back(pack_time);
    time_vector.push_back(gemm1_time);
    time_vector.push_back(gemm2_time);
    time_vector.push_back(dequantize_time);

    return std::make_pair(output_final, time_vector);
}