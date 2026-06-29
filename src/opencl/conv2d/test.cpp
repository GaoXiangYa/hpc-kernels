#include "test_utils.h"
#include "conv2d.h"
#include <vector>

static void conv2d_ref(const float* input, const float* kernel, float* output,
                       int input_rows, int input_cols, int kernel_rows,
                       int kernel_cols) {
  int out_rows = input_rows - kernel_rows + 1;
  int out_cols = input_cols - kernel_cols + 1;
  for (int i = 0; i < out_rows; ++i) {
    for (int j = 0; j < out_cols; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < kernel_rows; ++r)
        for (int c = 0; c < kernel_cols; ++c)
          sum += input[(i + r) * input_cols + (j + c)] *
                 kernel[r * kernel_cols + c];
      output[i * out_cols + j] = sum;
    }
  }
}

using Conv2DFunc = void (*)(const float*, const float*, float*, int, int, int,
                            int);

struct Conv2DVariant {
  const char* name;
  Conv2DFunc  func;
};

class Conv2DTest : public ::testing::TestWithParam<Conv2DVariant> {
protected:
  static constexpr int input_rows  = 3072;
  static constexpr int input_cols  = 3072;
  static constexpr int kernel_rows = 15;
  static constexpr int kernel_cols = 15;
  static constexpr int output_rows = input_rows - kernel_rows + 1;
  static constexpr int output_cols = input_cols - kernel_cols + 1;

  std::vector<float> input, filter, output_cpu, output_ocl;

  void SetUp() override {
    input      = random_vec(input_rows * input_cols);
    filter     = random_vec(kernel_rows * kernel_cols);
    output_cpu.assign(output_rows * output_cols, 0.0f);
    output_ocl.assign(output_rows * output_cols, 0.0f);
  }
};

TEST_P(Conv2DTest, Correctness) {
  auto [name, func] = GetParam();
  conv2d_ref(input.data(), filter.data(), output_cpu.data(), input_rows,
             input_cols, kernel_rows, kernel_cols);
  func(input.data(), filter.data(), output_ocl.data(), input_rows, input_cols,
       kernel_rows, kernel_cols);
  expect_near(output_ocl, output_cpu);
}

INSTANTIATE_TEST_SUITE_P(
    Variants, Conv2DTest,
    ::testing::Values(Conv2DVariant{"v0", valid_conv2d_v0},
                      Conv2DVariant{"v1", valid_conv2d_v1},
                      Conv2DVariant{"v2", valid_conv2d_v2},
                      Conv2DVariant{"v3", valid_conv2d_v3},
                      Conv2DVariant{"v4", valid_conv2d_v4}),
    [](const auto& info) { return info.param.name; });
