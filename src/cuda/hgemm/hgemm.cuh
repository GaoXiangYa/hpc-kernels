#pragma once

#include <cuda_bf16.h>

#define GEMM_A(i, j) A[(i) * lda + (j)]
#define GEMM_B(i, j) B[(i) * ldb + (j)]
#define GEMM_C(i, j) C[(i) * ldc + (j)]

void hgemm_v0(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c,
              int M, int N, int K);