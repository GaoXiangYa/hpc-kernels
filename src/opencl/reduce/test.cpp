#include "test_utils.h"
#include "reduce.h"
#include <numeric>
#include <vector>

// CPU reference
static float reduce_ref(const std::vector<float> &input) {
  return std::accumulate(input.begin(), input.end(), 0.0f);
}

// Variant descriptor
using ReduceFunc = void (*)(const float *, float *, int);
struct ReduceVariant {
  const char *name;
  ReduceFunc func;
};

class ReduceTest : public ::testing::TestWithParam<ReduceVariant> {
protected:
  static constexpr int kN = 4096;
  std::vector<float> input;
  float ref_output;

  void SetUp() override {
    input = random_vec(kN, -1.0f, 1.0f);
    ref_output = reduce_ref(input);
  }
};

TEST_P(ReduceTest, Correctness) {
  auto [name, func] = GetParam();
  float ocl_output = 0.0f;
  func(input.data(), &ocl_output, kN);
  EXPECT_NEAR(ocl_output, ref_output, 1e-3f);
}

INSTANTIATE_TEST_SUITE_P(
    Variants, ReduceTest,
    ::testing::Values(ReduceVariant{"v0", reduce_v0},
                      ReduceVariant{"v1", reduce_v1},
                      ReduceVariant{"v2", reduce_v2},
                      ReduceVariant{"v3", reduce_v3},
                      ReduceVariant{"v4", reduce_v4}),
    [](const auto &info) { return info.param.name; });
