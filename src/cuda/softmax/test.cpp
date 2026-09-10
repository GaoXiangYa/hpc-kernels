#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "softmax.h"
#include "util.h"

static inline void softmax_ref(const std::vector<float>& input,
                               std::vector<float>& output) {
  float max_ele = *std::max_element(input.begin(), input.end());
  float sum = 0.0f;
  for (auto val : input) {
    sum += std::exp(val - max_ele);
  }
  for (int i = 0; i < input.size(); ++i) {
    output[i] = std::exp(input[i] - max_ele) / sum;
  }
}

TEST(softmax, softmax_v0) {
  constexpr int kInputLen = 1024;
  std::vector<float> input(kInputLen, 0.0f);
  std::vector<float> cpu_output(kInputLen, 0.0f);
  std::vector<float> cuda_output(kInputLen, 0.0f);

  init_random(input, -1.0f, 1.0f);

  softmax_ref(input, cpu_output);

  softmax_v0(input.data(), cuda_output.data(), kInputLen);

  constexpr float kEpision = 1e-3f;
  for (int i = 0; i < cpu_output.size(); ++i) {
    ASSERT_NEAR(cpu_output[i], cuda_output[i], kEpision);
  }
}

TEST(softmax, softmax_v1) {
  constexpr int kInputLen = 1024;
  std::vector<float> input(kInputLen, 0.0f);
  std::vector<float> cpu_output(kInputLen, 0.0f);
  std::vector<float> cuda_output(kInputLen, 0.0f);

  init_random(input, -1.0f, 1.0f);

  softmax_ref(input, cpu_output);

  softmax_v1(input.data(), cuda_output.data(), kInputLen);

  constexpr float kEpision = 1e-3f;
  for (int i = 0; i < cpu_output.size(); ++i) {
    ASSERT_NEAR(cpu_output[i], cuda_output[i], kEpision);
  }
}

TEST(softmax, softmax_v2) {
  constexpr int kInputLen = 1024;
  std::vector<float> input(kInputLen, 0.0f);
  std::vector<float> cpu_output(kInputLen, 0.0f);
  std::vector<float> cuda_output(kInputLen, 0.0f);

  init_random(input, -1.0f, 1.0f);

  softmax_ref(input, cpu_output);

  softmax_v2(input.data(), cuda_output.data(), kInputLen);

  constexpr float kEpision = 1e-3f;
  for (int i = 0; i < cpu_output.size(); ++i) {
    ASSERT_NEAR(cpu_output[i], cuda_output[i], kEpision);
  }
}