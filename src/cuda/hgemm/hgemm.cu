#include "hgemm.cuh"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "hgemm_v0_kernel.cuh"
#include "util.h"

using Config = HgemmV0Config;

static void check_hgemm_v0_shape(int M, int N, int K) {
  if (M % Config::kBlockM != 0 || N % Config::kBlockN != 0 ||
      K % Config::kBlockK != 0) {
    std::fprintf(stderr,
                 "hgemm_v0: M/N/K must be multiples of %d/%d/%d, got %d/%d/%d\n",
                 Config::kBlockM, Config::kBlockN, Config::kBlockK, M, N, K);
    std::exit(EXIT_FAILURE);
  }
}

void hgemm_v0(const __nv_bfloat16* a, const __nv_bfloat16* b,
              __nv_bfloat16* c, int M, int N, int K) {
  check_hgemm_v0_shape(M, N, K);

  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  const size_t size_a = (size_t)M * lda * sizeof(__nv_bfloat16);
  const size_t size_b = (size_t)K * ldb * sizeof(__nv_bfloat16);
  const size_t size_c = (size_t)M * ldc * sizeof(__nv_bfloat16);

  __nv_bfloat16* dev_a = nullptr;
  __nv_bfloat16* dev_b = nullptr;
  __nv_bfloat16* dev_c = nullptr;

  CHECK_CUDA(cudaMalloc(&dev_a, size_a));
  CHECK_CUDA(cudaMalloc(&dev_b, size_b));
  CHECK_CUDA(cudaMalloc(&dev_c, size_c));

  CHECK_CUDA(cudaMemcpy(dev_a, a, size_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, b, size_b, cudaMemcpyHostToDevice));

  constexpr int kSmemElems =
      Config::kBlockM * Config::kSmemStrideA +
      Config::kBlockK * Config::kSmemStrideB;
  constexpr int kSmemBytes = kSmemElems * sizeof(__nv_bfloat16);

  dim3 block(Config::kThreadsPerBlock);
  dim3 grid(N / Config::kBlockN, M / Config::kBlockM);

  hgemm_v0_kernel<Config><<<grid, block, kSmemBytes>>>(
      dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}
