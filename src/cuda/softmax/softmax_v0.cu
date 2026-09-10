#include "reduce.cuh"
#include "util.h"

#define WARP_SIZE 32

// native softmax, each block calculate one token
static __global__ void softmax_v0_kernel(const float* input, float* output,
                                         const int n) {
  const int tx = threadIdx.x;
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= n) {
    return;
  }

  float max_num = 0.0f;
  max_num = blockDim.x <= 32 ? warpReduceMax(max_num) : blockReduceMax(max_num);

  __shared__ float s_max, s_sum;
  if (tx == 0) {
    s_max = max_num;
  }
  __syncthreads();

  float tmp = static_cast<float>(__ldg(input + idx));
  float sum_num = exp(tmp - max_num);
  sum_num = blockDim.x <= 32 ? warpReduceSum(sum_num) : blockReduceSum(sum_num);

  if (tx == 0) {
    s_sum = sum_num;
  }
  __syncthreads();

  output[idx] = exp(tmp - s_max) / s_sum;
}

void softmax_v0(float* input, float* output, int n) {
  float* input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, n * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(input_dev, input, n * sizeof(float), cudaMemcpyHostToDevice));

  float* output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(output_dev, output, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  constexpr int kThreadPerBlock = 1024;
  constexpr int kBlockPerGrid = 1;
  dim3 grid(kBlockPerGrid);
  dim3 block(kThreadPerBlock);

  softmax_v0_kernel<<<grid, block>>>(input_dev, output_dev, n);

  CHECK_CUDA(cudaMemcpy(output, output_dev, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
}
