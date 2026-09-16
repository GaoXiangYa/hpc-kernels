#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "hgemm_v0_kernel.cuh"
#include "util.h"

using Config = HgemmV0Config;

// ---------------------------------------------------------------------------
// Kernel launch helpers
// ---------------------------------------------------------------------------

static void launch_hgemm_v0(const __nv_bfloat16* da, const __nv_bfloat16* db,
                            __nv_bfloat16* dc, int M, int N, int K) {
  constexpr int kSmemElems =
      Config::kBlockM * Config::kSmemStrideA +
      Config::kBlockK * Config::kSmemStrideB;
  constexpr int kSmemBytes = kSmemElems * sizeof(__nv_bfloat16);

  dim3 block(Config::kThreadsPerBlock);
  dim3 grid(N / Config::kBlockN, M / Config::kBlockM);

  hgemm_v0_kernel<Config><<<grid, block, kSmemBytes>>>(da, db, dc, M, N, K, K,
                                                       N, N);
}

// ---------------------------------------------------------------------------
// cuBLAS reference: C[M,N] = A[M,K] * B[K,N], row-major.
// cuBLAS is column-major, so compute C^T = B^T * A^T.
// ---------------------------------------------------------------------------

static cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    cublasCreate(&handle);
  }
  return handle;
}

static void launch_cublas(const __nv_bfloat16* da, const __nv_bfloat16* db,
                          __nv_bfloat16* dc, int M, int N, int K) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasGemmEx(get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
               db, CUDA_R_16BF, N, da, CUDA_R_16BF, K, &beta, dc, CUDA_R_16BF,
               N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

static double bench(const char* name,
                    void (*launch)(const __nv_bfloat16*, const __nv_bfloat16*,
                                   __nv_bfloat16*, int, int, int),
                    const __nv_bfloat16* da, const __nv_bfloat16* db,
                    __nv_bfloat16* dc, int M, int N, int K) {
  launch(da, db, dc, M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const long long flops = 2LL * M * N * K;
  const int repeat = flops > 2LL * 1024 * 1024 * 1024 ? 3 : 10;

  cudaEventRecord(start);
  for (int r = 0; r < repeat; ++r) {
    launch(da, db, dc, M, N, K);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= static_cast<float>(repeat);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const double gflops =
      static_cast<double>(flops) / (static_cast<double>(ms) / 1000.0) / 1e9;

  std::printf("%s,%d,%d,%d,%.2f\n", name, M, N, K, gflops);
  return gflops;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::printf("version,M,N,K,gflops\n");

  const int begin = 256;
  const int end = 2048;
  const int step = 256;

  for (int size = begin; size <= end; size += step) {
    const int M = size;
    const int N = size;
    const int K = size;

    std::vector<__nv_bfloat16> ha(M * K);
    std::vector<__nv_bfloat16> hb(K * N);
    for (auto& v : ha) {
      v = __float2bfloat16(static_cast<float>(std::rand()) / RAND_MAX * 2.0f -
                           1.0f);
    }
    for (auto& v : hb) {
      v = __float2bfloat16(static_cast<float>(std::rand()) / RAND_MAX * 2.0f -
                           1.0f);
    }

    __nv_bfloat16* da = nullptr;
    __nv_bfloat16* db = nullptr;
    __nv_bfloat16* dc = nullptr;

    CHECK_CUDA(cudaMalloc(&da, (size_t)M * K * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&db, (size_t)K * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dc, (size_t)M * N * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(da, ha.data(), (size_t)M * K * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, hb.data(), (size_t)K * N * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    bench("hgemm_v0", launch_hgemm_v0, da, db, dc, M, N, K);
    bench("cublas", launch_cublas, da, db, dc, M, N, K);

    CHECK_CUDA(cudaFree(da));
    CHECK_CUDA(cudaFree(db));
    CHECK_CUDA(cudaFree(dc));
  }

  return 0;
}
