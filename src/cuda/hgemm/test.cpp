#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "hgemm.cuh"

using HgemmKernel =
    void (*)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int,
             int, int);

// ---------------------------------------------------------------------------
// Reference: BF16 inputs, FP32 accumulation.
// ---------------------------------------------------------------------------
static void ref_hgemm(const std::vector<__nv_bfloat16>& a,
                      const std::vector<__nv_bfloat16>& b,
                      std::vector<float>& c,
                      int M,
                      int N,
                      int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += __bfloat162float(a[i * K + k]) *
               __bfloat162float(b[k * N + j]);
      }
      c[i * N + j] = sum;
    }
  }
}

// ---------------------------------------------------------------------------
// Single test helper.
// ---------------------------------------------------------------------------
static void test_hgemm(HgemmKernel kernel, int M, int N, int K) {
  std::vector<__nv_bfloat16> a(M * K);
  std::vector<__nv_bfloat16> b(K * N);
  std::vector<__nv_bfloat16> c_cuda(M * N);
  std::vector<float> c_ref(M * N, 0.0f);
  std::vector<float> c_float(M * N, 0.0f);

  for (auto& v : a) {
    v = __float2bfloat16(static_cast<float>(std::rand()) / RAND_MAX * 2.0f -
                         1.0f);
  }
  for (auto& v : b) {
    v = __float2bfloat16(static_cast<float>(std::rand()) / RAND_MAX * 2.0f -
                         1.0f);
  }

  ref_hgemm(a, b, c_ref, M, N, K);
  kernel(a.data(), b.data(), c_cuda.data(), M, N, K);

  for (int i = 0; i < M * N; ++i) {
    c_float[i] = __bfloat162float(c_cuda[i]);

    // BF16 output quantization + FP32 accumulation:
    // use a relative tolerance.
    const float tolerance = 1e-2f * std::max(1.0f, std::abs(c_ref[i]));
    ASSERT_NEAR(c_ref[i], c_float[i], tolerance)
        << "Mismatch at index " << i << ", M=" << M << ", N=" << N
        << ", K=" << K;
  }
}

// ---------------------------------------------------------------------------
// Test cases. All shapes are multiples of the v0 tile configuration.
// ---------------------------------------------------------------------------
#define HGEMM_TEST(kernel, name, M, N, K) \
  TEST(HgemmTest, name) {                 \
    test_hgemm(kernel, M, N, K);          \
  }

HGEMM_TEST(hgemm_v0, v0_square_small, 64, 32, 64)
HGEMM_TEST(hgemm_v0, v0_square_medium, 256, 256, 256)
HGEMM_TEST(hgemm_v0, v0_rect, 1024, 256, 128)
HGEMM_TEST(hgemm_v0, v0_skinny, 1024, 16, 256)

int main(int argc, char** argv) {
  std::srand(0);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
