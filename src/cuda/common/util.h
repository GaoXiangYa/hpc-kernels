#pragma once

#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define MAX(a, b) a < b ? b : a
#define MIN(a, b) a < b ? a : b

template <typename T>
void init_random(std::vector<T> &vec,
                 T min = std::is_integral<T>::value ? 0 : T(0.0),
                 T max = std::is_integral<T>::value ? 100 : T(1.0)) {
  static_assert(std::is_arithmetic<T>::value,
                "Matrix elements must be numeric types");

  std::random_device rd;
  std::mt19937 rng(rd());

  // 根据类型选择分布
  if constexpr (std::is_integral<T>::value) {
    std::uniform_int_distribution<T> dist(min, max);
    for (auto &element : vec) {
      element = dist(rng);
    }
  } else {
    std::uniform_real_distribution<T> dist(min, max);
    for (auto &element : vec) {
      element = dist(rng);
    }
  }
}
