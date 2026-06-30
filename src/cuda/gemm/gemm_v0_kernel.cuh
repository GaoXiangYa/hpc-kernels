#include "gemm.cuh"

// C[M, N] = A[M, K] * B[K, N]
// BM, BN: block tile size (threads per block in x, y)
// Each thread computes one output element
template <int BM, int BN, int BK>
__global__ void gemm_v0_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               int lda, int ldb, int ldc) {
  const int col = blockIdx.x * BM + threadIdx.x;
  const int row = blockIdx.y * BN + threadIdx.y;
  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += GEMM_A(row, k) * GEMM_B(k, col);
  }

  GEMM_C(row, col) = sum;
}
