#include "device.h"

#include <iostream>
#include <vector>

DeviceManager& DeviceManager::get() {
    static DeviceManager instance;
    return instance;
}

DeviceManager::DeviceManager() {
    context_ = cl::Context::getDefault();
    std::vector<cl::Device> devices =
        context_.getInfo<CL_CONTEXT_DEVICES>();
    device_ = devices[0];
    queue_ = cl::CommandQueue(context_, device_,
                              CL_QUEUE_PROFILING_ENABLE);
}

cl::Kernel DeviceManager::build_kernel(const std::string& cl_path,
                                       const std::string& kernel_name,
                                       const std::string& build_opts) {
    std::string src = read_file(cl_path);
    cl::Program program(context_, src);
    program.build({device_}, build_opts.c_str());
    return cl::Kernel(program, kernel_name.c_str());
}

cl::Buffer DeviceManager::create_ro_buffer(size_t bytes, const void* data) {
    return cl::Buffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      bytes, const_cast<void*>(data));
}

cl::Buffer DeviceManager::create_rw_buffer(size_t bytes, void* data) {
    return cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      bytes, data);
}

void DeviceManager::launch(const cl::Kernel& kernel,
                           const cl::NDRange& global,
                           const cl::NDRange& local,
                           const std::string& label) {
    cl::Event event;
    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
                                nullptr, &event);

    cl_ulong start =
        event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end =
        event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double ms = static_cast<double>(end - start) / 1e6;
    std::cout << std::format("Kernel {} execution time: {} ms\n",
                             label, ms);
}

void DeviceManager::read_buffer(const cl::Buffer& buf, size_t bytes,
                                void* dst) {
    queue_.enqueueReadBuffer(buf, CL_TRUE, 0, bytes, dst);
}
