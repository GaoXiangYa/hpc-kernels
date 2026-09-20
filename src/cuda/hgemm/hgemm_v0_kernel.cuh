#pragma once

#include <cuda_bf16.h>

#include "hgemm_config.cuh"

template <int Row, int Col, int Ld, class Config>
__device__ __forceinline__ void copy_tile(const __nv_bfloat16* __restrict__ src,
                                          const int src_ld,
                                          __nv_bfloat16* dst) {
  const int tid = threadIdx.x;
  constexpr int kThreads = Config::kThreadsPerBlock;
  constexpr int kTileSize = Row * Col;
  constexpr int kIter = (kTileSize + kThreads - 1) / kThreads;

#pragma unroll
  for (int i = 0; i < kIter; ++i) {
    const int idx = tid + i * kThreads;
    if ((kTileSize % kThreads) != 0 && (idx >= kTileSize))
      break;
    const int row = idx / Col;
    const int col = idx % Col;

    dst[row * Ld + col] = src[row * src_ld + col];
  }
}

// Per-thread warp tile -> register tile -> outer product accumulation.
// ============================================================================
template <class Config>
__device__ __forceinline__ void
mma(const __nv_bfloat16* As, const __nv_bfloat16* Bs,
    const ThreadCoord<Config>& thread, RegisterTile<float, Config>& acc) {
  __nv_bfloat16 reg_a[Config::kThreadM];
  __nv_bfloat16 reg_b[Config::kThreadN];

  for (int ik = 0; ik < Config::kBlockK; ++ik) {
#pragma unroll
    for (int m = 0; m < Config::kThreadM; ++m) {
      reg_a[m] = As[(thread.tile_m + m) * Config::kSmemStrideA + ik];
    }

#pragma unroll
    for (int n = 0; n < Config::kThreadN; ++n) {
      reg_b[n] = Bs[ik * Config::kSmemStrideB + thread.tile_n + n];
    }

#pragma unroll
    for (int m = 0; m < Config::kThreadM; ++m) {
#pragma unroll
      for (int n = 0; n < Config::kThreadN; ++n) {
        acc(m, n) += __bfloat162float(reg_a[m]) * __bfloat162float(reg_b[n]);
      }
    }
  }
}

template <class Config>
__device__ __forceinline__ void
store_C(__nv_bfloat16* __restrict__ C, const ThreadCoord<Config>& thread,
        const RegisterTile<float, Config>& acc, int ldc) {
#pragma unroll
  for (int i = 0; i < Config::kThreadM; ++i) {
#pragma unroll
    for (int j = 0; j < Config::kThreadN; ++j) {
      C[(thread.global_m + i) * ldc + thread.global_n + j] =
          __float2bfloat16(acc(i, j));
    }
  }
}

template <class Config>
__global__ void hgemm_v0_kernel(const __nv_bfloat16* __restrict__ A,
                                const __nv_bfloat16* __restrict__ B,
                                __nv_bfloat16* __restrict__ C, int M, int N,
                                int K, int lda, int ldb, int ldc) {
  BlockCoord<Config> block;
  WarpCoord<Config> warp(block);
  ThreadCoord<Config> thread(warp);

  RegisterTile<float, Config> acc;
  acc.clear();

  extern __shared__ __nv_bfloat16 shmem[];
  __nv_bfloat16* As = shmem;
  __nv_bfloat16* Bs = As + Config::kBlockM * Config::kSmemStrideA;

  for (int k0 = 0; k0 < K; k0 += Config::kBlockK) {
    copy_tile<Config::kBlockM, Config::kBlockK, Config::kSmemStrideA, Config>(
        A + block.tile_m * lda + k0, lda, As);

    copy_tile<Config::kBlockK, Config::kBlockN, Config::kSmemStrideB, Config>(
        B + k0 * ldb + block.tile_n, ldb, Bs);
    __syncthreads();

    mma<Config>(As, Bs, thread, acc);

    __syncthreads();
  }

  store_C<Config>(C, thread, acc, ldc);
}
