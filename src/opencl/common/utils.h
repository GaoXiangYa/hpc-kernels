#pragma once

#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <cstddef>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

inline std::string read_file(const std::string& path) {
  // 打开文件
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs) {
    std::cerr << "Error: Could not open file " << path << std::endl;
    return "";
  }

  // 获取文件的大小
  ifs.seekg(0, std::ios::end);
  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // 如果文件为空，直接返回空字符串
  if (size == 0) {
    return "";
  }

  // 读取文件内容
  std::string text(size, '\0');  // 分配足够的空间来存储文件内容
  if (ifs.read(&text[0], size)) {
    return text;
  } else {
    std::cerr << "Error: Failed to read the file " << path << std::endl;
    return "";
  }
}

template <typename T>
void set_random_values(std::vector<T>& input, T min, T max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(min, max);

  for (auto& num : input) {
    num = dis(gen);
  }
}

inline void print_opencl_limits(cl_device_id device, cl_kernel kernel) {
  size_t max_wg_size;
  size_t max_item_sizes[3];
  cl_uint max_dims;

  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                  &max_wg_size, NULL);

  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_item_sizes),
                  max_item_sizes, NULL);

  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
                  &max_dims, NULL);

  printf("Device limits:\n");
  printf("  Max work-group size: %zu\n", max_wg_size);
  printf("  Max work-item sizes: [%zu, %zu, %zu]\n", max_item_sizes[0],
         max_item_sizes[1], max_item_sizes[2]);
  printf("  Max dimensions: %u\n", max_dims);

  if (kernel) {
    size_t kernel_wg_size;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &kernel_wg_size, NULL);

    printf("Kernel limits:\n");
    printf("  Max kernel work-group size: %zu\n", kernel_wg_size);
  }
}

inline void print_kernel_profiling_info(const char* kernel_name,
                                        const cl::Event& event) {
  cl_ulong time_start, time_end;

  auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  double nanoSeconds = static_cast<double>(end - start);

  std::cout << std::format("Kernel {} execution time: {} ms\n", kernel_name,
                           nanoSeconds / 1e6);
}

class OCLKernel {
public:
  explicit OCLKernel(const std::string& source_path,
                     const std::string& kernel_name,
                     const std::string& build_options = ""): source_path_(source_path),
                                                               kernel_name_(kernel_name) {
    context_ = cl::Context::getDefault();
    std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device device = devices[0];

    queue_ = std::make_unique<cl::CommandQueue>(context_, device, CL_QUEUE_PROFILING_ENABLE);
    const std::string kernel_src = read_file(source_path);
    cl::Program program(context_, kernel_src);
    program.build({device}, build_options.c_str());
    kernel_ = cl::Kernel(program, kernel_name.c_str());
  }

  const cl::Context& GetKernelContext() const { return context_; }

  const std::unique_ptr<cl::CommandQueue>& GetCommandQueue() const { return queue_; }

  const cl::Kernel& GetKernel() const { return kernel_; }

  cl::Kernel& GetKernel() { return kernel_; }
  
  void ProflingKernel(const cl::Event& event) {
    print_kernel_profiling_info(kernel_name_.c_str(), event);
  }
  // Specialize the template for more complex arguments if needed
  template <typename T>
  void set_kernel_args(const T& arg, size_t index) {
    kernel_.setArg(index, arg);
  }

  template <typename... Args>
  void set_kernel_args(size_t index, Args&&... args) {
    (set_kernel_args(std::forward<Args>(args), index++), ...);
  }

private:
  std::string source_path_;
  std::string kernel_name_;
  cl::Kernel kernel_;
  cl::Context context_;
  std::unique_ptr<cl::CommandQueue> queue_;
};
