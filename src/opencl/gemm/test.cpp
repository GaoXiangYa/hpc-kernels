#include "test_utils.h"
#include "gemm.h"
#include <vector>

// CPU reference
static void gemm_ref(const float* A, const float* B, float* C, int M, int N,
                     int K, float alpha = 1.0f, float beta = 0.0f) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k)
        sum += A[m * K + k] * B[k * N + n];
      C[m * N + n] = alpha * sum + beta * C[m * N + n];
    }
  }
}

using GEMMFunc = void (*)(const float*, const float*, float*, int, int, int,
                          float, float);

struct GEMMVariant {
  const char* name;
  GEMMFunc    func;
  float       alpha = 1.0f;
  float       beta  = 0.0f;
};

class GEMMTest : public ::testing::TestWithParam<GEMMVariant> {
protected:
  static constexpr int M = 4096, N = 1024, K = 2048;
  std::vector<float> A, B, C_cpu, C_ocl;

  void SetUp() override {
    A = random_vec(M * K);
    B = random_vec(K * N);
    C_cpu.assign(M * N, 0.0f);
    C_ocl.assign(M * N, 0.0f);
  }
};

TEST_P(GEMMTest, Correctness) {
  auto [name, func, alpha, beta] = GetParam();
  gemm_ref(A.data(), B.data(), C_cpu.data(), M, N, K, alpha, beta);
  func(A.data(), B.data(), C_ocl.data(), M, N, K, alpha, beta);
  expect_near(C_ocl, C_cpu, 1e-3f);
}

INSTANTIATE_TEST_SUITE_P(
    Variants, GEMMTest,
    ::testing::Values(
        GEMMVariant{"v0", gemm_v0, 1.0f, 0.0f},
        GEMMVariant{"v1", gemm_v1, 1.0f, 0.0f},
        GEMMVariant{"v2", gemm_v2, 1.0f, 0.0f},
        GEMMVariant{"v3", gemm_v3, 1.0f, 0.0f},
        GEMMVariant{"v4", gemm_v4, 1.0f, 0.0f},
        GEMMVariant{"v5", gemm_v5, -1.0f, 0.1f},
        GEMMVariant{"v6", gemm_v6, -1.0f, 0.1f},
        GEMMVariant{"v7", gemm_v7, 1.0f, 0.1f},
        GEMMVariant{"v8", gemm_v8, 1.0f, 0.1f},
        GEMMVariant{"v9", gemm_v9, 1.0f, 0.1f},
        GEMMVariant{"v10", gemm_v10, 1.0f, 0.0f}),
    [](const auto& info) { return info.param.name; });
