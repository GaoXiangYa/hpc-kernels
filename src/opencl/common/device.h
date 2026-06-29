#pragma once

#include <CL/cl.h>
#include <CL/cl2.hpp>
#include <CL/opencl.hpp>

#include <format>
#include <fstream>
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// Utility: read a file into a string
// ---------------------------------------------------------------------------
inline std::string read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Could not open file " << path << std::endl;
        return "";
    }
    ifs.seekg(0, std::ios::end);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (size == 0) return "";
    std::string text(size, '\0');
    if (!ifs.read(&text[0], size)) {
        std::cerr << "Error: Failed to read file " << path << std::endl;
        return "";
    }
    return text;
}

// ---------------------------------------------------------------------------
// DeviceManager – singleton owning the OpenCL context, command queue and
// device.  Provides buffer factories, kernel building and launch helpers.
// ---------------------------------------------------------------------------
class DeviceManager {
public:
    static DeviceManager& get();

    cl::Context&       context() { return context_; }
    cl::CommandQueue&  queue()   { return queue_; }
    cl::Device&        device()  { return device_; }

    // Build a kernel from a .cl source file.
    cl::Kernel build_kernel(const std::string& cl_path,
                            const std::string& kernel_name,
                            const std::string& build_opts = "");

    // Buffer factories – eliminate CL_MEM_* noise.
    cl::Buffer create_ro_buffer(size_t bytes, const void* data);
    cl::Buffer create_rw_buffer(size_t bytes, void* data);

    // Enqueue a kernel, then print profiling information.
    void launch(const cl::Kernel& kernel,
                const cl::NDRange& global,
                const cl::NDRange& local,
                const std::string& label);

    // Blocking read from a device buffer back to host.
    void read_buffer(const cl::Buffer& buf, size_t bytes, void* dst);

private:
    DeviceManager();
    cl::Context      context_;
    cl::CommandQueue queue_;
    cl::Device       device_;
};
