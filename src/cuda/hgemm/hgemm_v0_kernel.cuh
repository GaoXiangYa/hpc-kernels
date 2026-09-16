#pragma once

#include <cuda_bf16.h>

#include "hgemm_config.cuh"

template <class Config>
__device__ __forceinline__ void load_A_to_tile(const __nv_bfloat16* __restrict__ A,
                                               __nv_bfloat16* As,
                                               const BlockCoord<Config>& block,
                                               const WarpCoord<Config>& warp,
                                               int lda,
                                               int k0) {
  ACopyCoord<Config> coord(block, warp, A, As, lda, k0);

#pragma unroll
  for (int i = 0; i < ACopyCoord<Config>::kIters; ++i) {
    coord.smem[0] = coord.gmem[0];
    coord.next();
  }
}

template <class Config>
__device__ __forceinline__ void load_B_to_tile(const __nv_bfloat16* __restrict__ B,
                                               __nv_bfloat16* Bs,
                                               const BlockCoord<Config>& block,
                                               const WarpCoord<Config>& warp,
                                               int ldb,
                                               int k0) {
  BCopyCoord<Config> coord(block, warp, B, Bs, ldb, k0);

#pragma unroll
  for (int i = 0; i < BCopyCoord<Config>::kIters; ++i) {
    coord.smem[0] = coord.gmem[0];
    coord.next();
  }
}

template <class Config>
__device__ __forceinline__ void mma(const __nv_bfloat16* As,
                                    const __nv_bfloat16* Bs,
                                    const ThreadCoord<Config>& thread,
                                    RegisterTile<float, Config>& acc) {
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
__device__ __forceinline__ void store_C(__nv_bfloat16* __restrict__ C,
                                        const ThreadCoord<Config>& thread,
                                        const RegisterTile<float, Config>& acc,
                                        int ldc) {
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
    load_A_to_tile<Config>(A, As, block, warp, lda, k0);
    load_B_to_tile<Config>(B, Bs, block, warp, ldb, k0);

    __syncthreads();

    mma<Config>(As, Bs, thread, acc);

    __syncthreads();
  }

  store_C<Config>(C, thread, acc, ldc);
}
