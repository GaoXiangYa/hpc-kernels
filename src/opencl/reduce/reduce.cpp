#include "reduce.h"
#include "utils.h"
#include <CL/cl.h>
#include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <vector>

void reduce_v0(const float *input, float *output, int n) {
  const std::string build_options = "";

  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  const std::string kernel_src = read_file("../src/opencl/reduce/reduce_v0.cl");
  cl::Program program(context, kernel_src);
  program.build({device}, build_options.c_str());
  cl::Kernel kernel(program, "reduce_v0_kernel");

  constexpr int kThreadsPerBlock = 256;
  const int kNumBlocks =
      (n + kThreadsPerBlock - 1) / kThreadsPerBlock * kThreadsPerBlock;

  std::vector<float> temp_output(kNumBlocks, 0.0f);

  cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * n, (void *)input);
  cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kNumBlocks,
                           (void *)temp_output.data());

  kernel.setArg(0, input_buffer);
  kernel.setArg(1, output_buffer);
  kernel.setArg(2, n);

  cl::NDRange global_work_size(kNumBlocks);
  cl::NDRange local_work_size(kThreadsPerBlock);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size,
                             local_work_size, nullptr, &event);
  queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float) * kNumBlocks,
                          temp_output.data());

  for (int i = 0; i < kNumBlocks; ++i) {
    *output += temp_output[i];
  }

  print_kernel_profiling_info("reduce_v0_kernel", event);
}

void reduce_v1(const float *input, float *output, int n) {
  const std::string build_options = "";

  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  const std::string kernel_src = read_file("../src/opencl/reduce/reduce_v1.cl");
  cl::Program program(context, kernel_src);
  program.build({device}, build_options.c_str());
  cl::Kernel kernel(program, "reduce_v1_kernel");

  constexpr int kThreadsPerBlock = 256;
  const int kNumBlocks =
      (n + kThreadsPerBlock - 1) / kThreadsPerBlock * kThreadsPerBlock;

  std::vector<float> temp_output(kNumBlocks, 0.0f);

  cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * n, (void *)input);
  cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kNumBlocks,
                           (void *)temp_output.data());

  kernel.setArg(0, input_buffer);
  kernel.setArg(1, output_buffer);
  kernel.setArg(2, n);
  kernel.setArg(3, kThreadsPerBlock);

  cl::NDRange global_work_size(kNumBlocks);
  cl::NDRange local_work_size(kThreadsPerBlock);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size,
                             local_work_size, nullptr, &event);
  queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float) * kNumBlocks,
                          temp_output.data());

  for (int i = 0; i < kNumBlocks; ++i) {
    *output += temp_output[i];
  }

  print_kernel_profiling_info("reduce_v1_kernel", event);
}

void reduce_v2(const float *input, float *output, int n) {
  const std::string build_options = "";

  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  const std::string kernel_src = read_file("../src/opencl/reduce/reduce_v2.cl");
  cl::Program program(context, kernel_src);
  program.build({device}, build_options.c_str());
  cl::Kernel kernel(program, "reduce_v2_kernel");

  constexpr int kLocalSize = 256;
  const int kGlobalSize =
      (n + kLocalSize * 8 - 1) / (kLocalSize * 8) * (kLocalSize * 8);

  std::vector<float> temp_output(kGlobalSize, 0.0f);

  cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * n, (void *)input);
  cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kGlobalSize,
                           (void *)temp_output.data());

  kernel.setArg(0, input_buffer);
  kernel.setArg(1, output_buffer);
  kernel.setArg(2, n);

  cl::NDRange global_work_size(kGlobalSize);
  cl::NDRange local_work_size(kLocalSize);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size,
                             local_work_size, nullptr, &event);
  queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0,
                          sizeof(float) * kGlobalSize, temp_output.data());

  for (int i = 0; i < kGlobalSize; ++i) {
    *output += temp_output[i];
  }

  print_kernel_profiling_info("reduce_v2_kernel", event);
}

void reduce_v3(const float *input, float *output, int n) {
  const std::string build_options = "";

  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  const std::string kernel_src = read_file("../src/opencl/reduce/reduce_v3.cl");
  cl::Program program(context, kernel_src);
  program.build({device}, build_options.c_str());
  cl::Kernel kernel(program, "reduce_v3_kernel");

  constexpr int kLocalSize = 256;
  const int kGlobalSize = n / 4;

  std::vector<float> temp_output(kGlobalSize, 0.0f);

  cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * n, (void *)input);
  cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kGlobalSize,
                           (void *)temp_output.data());

  kernel.setArg(0, input_buffer);
  kernel.setArg(1, output_buffer);
  kernel.setArg(2, n);

  cl::NDRange global_work_size(kGlobalSize);
  cl::NDRange local_work_size(kLocalSize);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size,
                             local_work_size, nullptr, &event);
  queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0,
                          sizeof(float) * kGlobalSize, temp_output.data());

  for (int i = 0; i < kGlobalSize; ++i) {
    *output += temp_output[i];
  }

  print_kernel_profiling_info("reduce_v3_kernel", event);
}

void reduce_v4(const float *input, float *output, int n) {
  const std::string build_options = "";

  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  const std::string kernel_src = read_file("../src/opencl/reduce/reduce_v4.cl");
  cl::Program program(context, kernel_src);
  program.build({device}, build_options.c_str());
  cl::Kernel kernel(program, "reduce_v4_kernel");

  constexpr int kLocalSize = 256;
  const int kGlobalSize = n / 4;

  std::vector<float> temp_output(kGlobalSize, 0.0f);

  cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * n, (void *)input);
  cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kGlobalSize,
                           (void *)temp_output.data());

  kernel.setArg(0, input_buffer);
  kernel.setArg(1, output_buffer);
  kernel.setArg(2, cl::Local(kLocalSize * sizeof(cl_float4) + 21));
  kernel.setArg(3, n);

  cl::NDRange global_work_size(kGlobalSize);
  cl::NDRange local_work_size(kLocalSize);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size,
                             local_work_size, nullptr, &event);
  queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0,
                          sizeof(float) * kGlobalSize, temp_output.data());

  for (int i = 0; i < kGlobalSize; ++i) {
    *output += temp_output[i];
  }
  print_kernel_profiling_info("reduce_v4_kernel", event);
}