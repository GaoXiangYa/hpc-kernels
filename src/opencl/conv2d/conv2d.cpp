#include "conv2d.h"
#include "device.h"
#include <algorithm>

void valid_conv2d_v0(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/conv2d/valid_conv2d_v0.cl",
                                "valid_conv2d_v0_kernel");

  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;

  cl::NDRange global_work_size(kOutputCols, kOutputRows);
  cl::NDRange local_work_size(16, 16);

  auto bi = dm.create_ro_buffer(sizeof(float) * input_rows * input_cols, input);
  auto bf =
      dm.create_ro_buffer(sizeof(float) * kernel_rows * kernel_cols, filter);
  auto bo = dm.create_rw_buffer(sizeof(float) * kOutputRows * kOutputCols,
                                output);

  kernel.setArg(0, bi);
  kernel.setArg(1, bf);
  kernel.setArg(2, bo);
  kernel.setArg(3, input_rows);
  kernel.setArg(4, input_cols);
  kernel.setArg(5, kernel_rows);
  kernel.setArg(6, kernel_cols);

  dm.launch(kernel, global_work_size, cl::NullRange,
            "valid_conv2d_v0_kernel");
  dm.read_buffer(bo, sizeof(float) * kOutputRows * kOutputCols, output);
}

void valid_conv2d_v1(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/conv2d/valid_conv2d_v1.cl",
                                "valid_conv2d_v1_kernel");

  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;

  cl::NDRange global_work_size(kOutputCols, kOutputRows);
  cl::NDRange local_work_size(16, 16);

  auto bi = dm.create_ro_buffer(sizeof(float) * input_rows * input_cols, input);
  auto bf =
      dm.create_ro_buffer(sizeof(float) * kernel_rows * kernel_cols, filter);
  auto bo = dm.create_rw_buffer(sizeof(float) * kOutputRows * kOutputCols,
                                output);

  kernel.setArg(0, bi);
  kernel.setArg(1, bf);
  kernel.setArg(2, bo);
  kernel.setArg(3, input_rows);
  kernel.setArg(4, input_cols);
  kernel.setArg(5, kernel_rows);
  kernel.setArg(6, kernel_cols);

  dm.launch(kernel, global_work_size, cl::NullRange,
            "valid_conv2d_v1_kernel");
  dm.read_buffer(bo, sizeof(float) * kOutputRows * kOutputCols, output);
}

void valid_conv2d_v2(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/conv2d/valid_conv2d_v2.cl",
                                "valid_conv2d_v2_kernel");

  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;

  const int kLocalSizeX = 16;
  const int kLocalSizeY = 16;

  const int kSharedInputSizeX = kLocalSizeX + kernel_cols - 1;
  const int kSharedInputSizeY = kLocalSizeY + kernel_rows - 1;

  cl::NDRange global_work_size(
      (kOutputCols + kLocalSizeX - 1) / kLocalSizeX * kLocalSizeX,
      (kOutputRows + kLocalSizeY - 1) / kLocalSizeY * kLocalSizeY);
  cl::NDRange local_work_size(kLocalSizeX, kLocalSizeY);

  auto bi = dm.create_ro_buffer(sizeof(float) * input_rows * input_cols, input);
  auto bf =
      dm.create_ro_buffer(sizeof(float) * kernel_rows * kernel_cols, filter);
  auto bo = dm.create_rw_buffer(sizeof(float) * kOutputRows * kOutputCols,
                                output);

  kernel.setArg(0, bi);
  kernel.setArg(1, bf);
  kernel.setArg(2, bo);
  kernel.setArg(
      3, cl::Local(sizeof(float) * (kSharedInputSizeX + 1) * kSharedInputSizeY));
  kernel.setArg(4, input_rows);
  kernel.setArg(5, input_cols);
  kernel.setArg(6, kernel_rows);
  kernel.setArg(7, kernel_cols);
  kernel.setArg(8, kOutputRows);
  kernel.setArg(9, kOutputCols);

  dm.launch(kernel, global_work_size, local_work_size,
            "valid_conv2d_v2_kernel");
  dm.read_buffer(bo, sizeof(float) * kOutputRows * kOutputCols, output);
}

void valid_conv2d_v3(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/conv2d/valid_conv2d_v3.cl",
                                "valid_conv2d_v3_kernel");

  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;
  const int kRegX = 2;
  const int kRegY = 2;
  const int kLocalSizeX = std::min(16, kOutputCols);
  const int kLocalSizeY = std::min(16, kOutputRows);
  const int kGlobalSizeX = (kOutputCols + (kLocalSizeX * kRegX) - 1) /
                           (kLocalSizeX * kRegX) * kLocalSizeX;
  const int kGlobalSizeY = (kOutputRows + (kLocalSizeY * kRegY) - 1) /
                           (kLocalSizeY * kRegY) * kLocalSizeY;

  cl::NDRange global_work_size(kGlobalSizeX, kGlobalSizeY);
  cl::NDRange local_work_size(kLocalSizeX, kLocalSizeY);

  auto bi = dm.create_ro_buffer(sizeof(float) * input_rows * input_cols, input);
  auto bf =
      dm.create_ro_buffer(sizeof(float) * kernel_rows * kernel_cols, filter);
  auto bo = dm.create_rw_buffer(sizeof(float) * kOutputRows * kOutputCols,
                                output);

  kernel.setArg(0, bi);
  kernel.setArg(1, bf);
  kernel.setArg(2, bo);
  kernel.setArg(3, input_rows);
  kernel.setArg(4, input_cols);
  kernel.setArg(5, kernel_rows);
  kernel.setArg(6, kernel_cols);
  kernel.setArg(7, kOutputRows);
  kernel.setArg(8, kOutputCols);

  dm.launch(kernel, global_work_size, local_work_size,
            "valid_conv2d_v3_kernel");
  dm.read_buffer(bo, sizeof(float) * kOutputRows * kOutputCols, output);
}

void valid_conv2d_v4(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/conv2d/valid_conv2d_v4.cl",
                                "valid_conv2d_v4_kernel");

  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;
  const int kRegX = 2;
  const int kRegY = 2;
  const int kLocalSizeX = std::min(16, kOutputCols);
  const int kLocalSizeY = std::min(16, kOutputRows);
  const int kSharedInputX = kLocalSizeX + kernel_cols - 1;
  const int kSharedInputY = kLocalSizeY + kernel_rows - 1;
  const int kGlobalSizeX = (kOutputCols + (kLocalSizeX * kRegX) - 1) /
                           (kLocalSizeX * kRegX) * kLocalSizeX;
  const int kGlobalSizeY = (kOutputRows + (kLocalSizeY * kRegY) - 1) /
                           (kLocalSizeY * kRegY) * kLocalSizeY;

  cl::NDRange global_work_size(kGlobalSizeX, kGlobalSizeY);
  cl::NDRange local_work_size(kLocalSizeX, kLocalSizeY);

  auto bi = dm.create_ro_buffer(sizeof(float) * input_rows * input_cols, input);
  auto bf =
      dm.create_ro_buffer(sizeof(float) * kernel_rows * kernel_cols, filter);
  auto bo = dm.create_rw_buffer(sizeof(float) * kOutputRows * kOutputCols,
                                output);

  kernel.setArg(0, bi);
  kernel.setArg(1, bf);
  kernel.setArg(2, bo);
  kernel.setArg(3, cl::Local(sizeof(float) * kRegX * kRegY * kSharedInputX *
                             kSharedInputY));
  kernel.setArg(4, input_rows);
  kernel.setArg(5, input_cols);
  kernel.setArg(6, kernel_rows);
  kernel.setArg(7, kernel_cols);
  kernel.setArg(8, kOutputRows);
  kernel.setArg(9, kOutputCols);

  dm.launch(kernel, global_work_size, local_work_size,
            "valid_conv2d_v4_kernel");
  dm.read_buffer(bo, sizeof(float) * kOutputRows * kOutputCols, output);
}
