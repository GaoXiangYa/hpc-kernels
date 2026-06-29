#include "reduce.h"
#include "device.h"
#include <CL/cl.h>
#include <vector>

void reduce_v0(const float *input, float *output, int n) {
  auto &dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/reduce/reduce_v0.cl",
                                "reduce_v0_kernel");

  constexpr int kThreadsPerBlock = 256;
  const int kNumBlocks =
      (n + kThreadsPerBlock - 1) / kThreadsPerBlock * kThreadsPerBlock;

  std::vector<float> temp_output(kNumBlocks, 0.0f);

  auto buf_in = dm.create_ro_buffer(sizeof(float) * n, input);
  auto buf_out =
      dm.create_rw_buffer(sizeof(float) * kNumBlocks, temp_output.data());

  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_out);
  kernel.setArg(2, n);

  cl::NDRange global_work_size(kNumBlocks);
  cl::NDRange local_work_size(kThreadsPerBlock);

  dm.launch(kernel, global_work_size, local_work_size, "reduce_v0_kernel");
  dm.read_buffer(buf_out, sizeof(float) * kNumBlocks, temp_output.data());

  for (int i = 0; i < kNumBlocks; ++i) {
    *output += temp_output[i];
  }
}

void reduce_v1(const float *input, float *output, int n) {
  auto &dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/reduce/reduce_v1.cl",
                                "reduce_v1_kernel");

  constexpr int kThreadsPerBlock = 256;
  const int kNumBlocks =
      (n + kThreadsPerBlock - 1) / kThreadsPerBlock * kThreadsPerBlock;

  std::vector<float> temp_output(kNumBlocks, 0.0f);

  auto buf_in = dm.create_ro_buffer(sizeof(float) * n, input);
  auto buf_out =
      dm.create_rw_buffer(sizeof(float) * kNumBlocks, temp_output.data());

  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_out);
  kernel.setArg(2, n);
  kernel.setArg(3, kThreadsPerBlock);

  cl::NDRange global_work_size(kNumBlocks);
  cl::NDRange local_work_size(kThreadsPerBlock);

  dm.launch(kernel, global_work_size, local_work_size, "reduce_v1_kernel");
  dm.read_buffer(buf_out, sizeof(float) * kNumBlocks, temp_output.data());

  for (int i = 0; i < kNumBlocks; ++i) {
    *output += temp_output[i];
  }
}

void reduce_v2(const float *input, float *output, int n) {
  auto &dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/reduce/reduce_v2.cl",
                                "reduce_v2_kernel");

  constexpr int kLocalSize = 256;
  const int kGlobalSize =
      (n + kLocalSize * 8 - 1) / (kLocalSize * 8) * (kLocalSize * 8);

  std::vector<float> temp_output(kGlobalSize, 0.0f);

  auto buf_in = dm.create_ro_buffer(sizeof(float) * n, input);
  auto buf_out =
      dm.create_rw_buffer(sizeof(float) * kGlobalSize, temp_output.data());

  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_out);
  kernel.setArg(2, n);

  cl::NDRange global_work_size(kGlobalSize);
  cl::NDRange local_work_size(kLocalSize);

  dm.launch(kernel, global_work_size, local_work_size, "reduce_v2_kernel");
  dm.read_buffer(buf_out, sizeof(float) * kGlobalSize, temp_output.data());

  for (int i = 0; i < kGlobalSize; ++i) {
    *output += temp_output[i];
  }
}

void reduce_v3(const float *input, float *output, int n) {
  auto &dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/reduce/reduce_v3.cl",
                                "reduce_v3_kernel");

  constexpr int kLocalSize = 256;
  const int kGlobalSize = n / 4;

  std::vector<float> temp_output(kGlobalSize, 0.0f);

  auto buf_in = dm.create_ro_buffer(sizeof(float) * n, input);
  auto buf_out =
      dm.create_rw_buffer(sizeof(float) * kGlobalSize, temp_output.data());

  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_out);
  kernel.setArg(2, n);

  cl::NDRange global_work_size(kGlobalSize);
  cl::NDRange local_work_size(kLocalSize);

  dm.launch(kernel, global_work_size, local_work_size, "reduce_v3_kernel");
  dm.read_buffer(buf_out, sizeof(float) * kGlobalSize, temp_output.data());

  for (int i = 0; i < kGlobalSize; ++i) {
    *output += temp_output[i];
  }
}

void reduce_v4(const float *input, float *output, int n) {
  auto &dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/reduce/reduce_v4.cl",
                                "reduce_v4_kernel");

  constexpr int kLocalSize = 256;
  const int kGlobalSize = n / 4;

  std::vector<float> temp_output(kGlobalSize, 0.0f);

  auto buf_in = dm.create_ro_buffer(sizeof(float) * n, input);
  auto buf_out =
      dm.create_rw_buffer(sizeof(float) * kGlobalSize, temp_output.data());

  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_out);
  kernel.setArg(2,
                cl::Local(kLocalSize * sizeof(cl_float4) + 21));
  kernel.setArg(3, n);

  cl::NDRange global_work_size(kGlobalSize);
  cl::NDRange local_work_size(kLocalSize);

  dm.launch(kernel, global_work_size, local_work_size, "reduce_v4_kernel");
  dm.read_buffer(buf_out, sizeof(float) * kGlobalSize, temp_output.data());

  for (int i = 0; i < kGlobalSize; ++i) {
    *output += temp_output[i];
  }
}
