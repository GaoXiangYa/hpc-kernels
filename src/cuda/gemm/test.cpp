#include "gemm.cuh"
#include "test_util.h"
#include <gtest/gtest.h>
#include <vector>

TEST(GemmTest, gemm_v0) {
  const int m = 1024, n = 256, k = 128;
  std::vector<float> host_a(m * n);
  std::vector<float> host_b(n * k);
  std::vector<float> host_cpu_c(m * k, 0.0f);
  std::vector<float> host_gpu_c(m * k, 0.0f);

  init_random_matrix(host_a, -1.0f, 1.0f);
  init_random_matrix(host_b, -1.0f, 1.0f);

  ref_matmul(host_a, host_b, host_cpu_c, m, n, k);
  gemm_v0(host_a.data(), host_b.data(), host_gpu_c.data(), m, n, k);

  compare_matrix(host_cpu_c, host_gpu_c);
}

TEST(GemmTest, gemm_v1) {
  const int m = 1024, n = 256, k = 128;
  // const int m = 4, n = 4, k = 4;
  std::vector<float> host_a(m * n);
  std::vector<float> host_b(n * k);
  std::vector<float> host_cpu_c(m * k, 0.0f);
  std::vector<float> host_gpu_c(m * k, 0.0f);

  init_random_matrix(host_a, 1.0f, 1.0f);
  init_random_matrix(host_b, 1.0f, 1.0f);

  ref_matmul(host_a, host_b, host_cpu_c, m, n, k);
  gemm_v1(host_a.data(), host_b.data(), host_gpu_c.data(), m, n, k); 

  // print_matmul(host_a.data(), m, n);
  // printf("======================\n");
  // print_matmul(host_b.data(), n, k);
  // print_matmul(host_gpu_c.data(), m, k);
  compare_matrix(host_cpu_c, host_gpu_c);
}

TEST(GemmTest, gemm_v2) {
  const int m = 1024, n = 256, k = 128;
  // const int m = 4, n = 4, k = 4;
  std::vector<float> host_a(m * n);
  std::vector<float> host_b(n * k);
  std::vector<float> host_cpu_c(m * k, 0.0f);
  std::vector<float> host_gpu_c(m * k, 0.0f);

  init_random_matrix(host_a, 1.0f, 1.0f);
  init_random_matrix(host_b, 1.0f, 1.0f);

  ref_matmul(host_a, host_b, host_cpu_c, m, n, k);
  gemm_v2(host_a.data(), host_b.data(), host_gpu_c.data(), m, n, k); 

  // print_matmul(host_a.data(), m, n);
  // printf("======================\n");
  // print_matmul(host_b.data(), n, k);
  // print_matmul(host_gpu_c.data(), m, k);
  compare_matrix(host_cpu_c, host_gpu_c);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  return 0;
}