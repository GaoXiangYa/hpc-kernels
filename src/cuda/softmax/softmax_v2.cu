#include "softmax.h"
#include "util.h"

#define WARP_SIZE 32

struct SoftmaxState {
  float m;
  float d;
};

__device__ __forceinline__ SoftmaxState softmaxMerge(const SoftmaxState& a,
                                                     const SoftmaxState& b) {
  if (a.m == -INFINITY) {
    return b;
  }
  if (b.m == -INFINITY) {
    return a;
  }
  SoftmaxState ret;
  ret.m = fmaxf(a.m, b.m);
  ret.d = exp(a.m - ret.m) * a.d + exp(b.m - ret.m) * b.d;
  return ret;
}

__device__ __forceinline__ SoftmaxState warpOnlineReduce(SoftmaxState val) {
  auto mask = __activemask();
#pragma unroll
  for (size_t offset = 16; offset >= 1; offset >>= 1) {
    SoftmaxState other;
    other.m = __shfl_down_sync(mask, val.m, offset);
    other.d = __shfl_down_sync(mask, val.d, offset);
    val = softmaxMerge(val, other);
  }
  return val;
}

template <int BLOCK_THREADS>
__device__ __forceinline__ SoftmaxState blockOnlineReduce(SoftmaxState val) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;
  const int warp_nums = BLOCK_THREADS / WARP_SIZE;
  __shared__ SoftmaxState sh_state[warp_nums];

  val = warpOnlineReduce(val);
  if (lane_id == 0) {
    sh_state[warp_id] = val;
  }
  __syncthreads();

  SoftmaxState block_value = {-INFINITY, 0.0f};
  if (warp_id == 0) {
    if (lane_id < warp_nums) {
      block_value = sh_state[lane_id];
    }
    block_value = warpOnlineReduce(block_value);
    if (lane_id == 0) {
      sh_state[0] = block_value;
    }
  }
  __syncthreads();

  return sh_state[0];
}

// online softmax

template <int BLOCK_THREADS>
static __global__ void softmax_v2_kernel(const float* input, float* output,
                                         int n) {
  const int tx = threadIdx.x;
  const int block_size = blockDim.x;

  SoftmaxState state;
  state.m = -INFINITY;
  state.d = 0.0f;
  // online softmax update
  for (int i = tx; i < n; i += block_size) {
    float val = input[i];
    float new_m = fmax(val, state.m);
    state.d = exp(state.m - new_m) * state.d + exp(val - new_m);
    state.m = new_m;
  }

  state = BLOCK_THREADS <= 32 ? warpOnlineReduce(state)
                              : blockOnlineReduce<BLOCK_THREADS>(state);

  float m = state.m;
  float d = 1.0f / state.d;

  for (int i = tx; i < n; i += block_size) {
    output[i] = exp(input[i] - m) * d;
  }
}

void softmax_v2(float* input, float* output, int n) {
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

  softmax_v2_kernel<kThreadPerBlock><<<grid, block>>>(input_dev, output_dev, n);

  CHECK_CUDA(cudaMemcpy(output, output_dev, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
}