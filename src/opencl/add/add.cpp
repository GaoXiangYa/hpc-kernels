#include "add.h"
#include "device.h"

// ---------------------------------------------------------------------------
// Raw-pointer interface
// ---------------------------------------------------------------------------
void launch_add_v0(const float* A, const long* strides_A, long size_A,
                   const float* B, const long* strides_B, long size_B,
                         float* C, const long* strides_C, long size_C,
                   const long* shape, int ndim, long total_elements) {

    auto& dm = DeviceManager::get();
    auto kernel = dm.build_kernel("../src/opencl/add/add_v0.cl",
                                  "add_kernel");

    // Data — allocate exactly the physical tensor sizes
    auto d_A = dm.create_ro_buffer(sizeof(float) * size_A, A);
    auto d_B = dm.create_ro_buffer(sizeof(float) * size_B, B);
    auto d_C = dm.create_rw_buffer(sizeof(float) * size_C, C);

    // Metadata (strides + shape)
    auto d_strides_A = dm.create_ro_buffer(sizeof(long) * ndim, strides_A);
    auto d_strides_B = dm.create_ro_buffer(sizeof(long) * ndim, strides_B);
    auto d_strides_C = dm.create_ro_buffer(sizeof(long) * ndim, strides_C);
    auto d_shape     = dm.create_ro_buffer(sizeof(long) * ndim, shape);

    kernel.setArg(0,  d_A);
    kernel.setArg(1,  d_strides_A);
    kernel.setArg(2,  d_B);
    kernel.setArg(3,  d_strides_B);
    kernel.setArg(4,  d_C);
    kernel.setArg(5,  d_strides_C);
    kernel.setArg(6,  d_shape);
    kernel.setArg(7,  ndim);
    kernel.setArg(8,  total_elements);

    dm.launch(kernel, cl::NDRange(total_elements), cl::NullRange, "add_kernel");
    dm.read_buffer(d_C, sizeof(float) * size_C, C);
}

// ---------------------------------------------------------------------------
// v1 ── dispatch by ndim
// ---------------------------------------------------------------------------
static const char* kernel_for_ndim(int ndim) {
    switch (ndim) {
        case 1:  return "add_kernel_dim1";
        case 2:  return "add_kernel_dim2";
        case 3:  return "add_kernel_dim3";
        default: return "add_kernel_dim4";  // ndim>=4
    }
}

void launch_add_v1(const float* A, const long* strides_A, long size_A,
                   const float* B, const long* strides_B, long size_B,
                         float* C, const long* strides_C, long size_C,
                   const long* shape, int ndim, long total_elements) {

    auto& dm = DeviceManager::get();
    auto kernel = dm.build_kernel("../src/opencl/add/add_v1.cl",
                                  kernel_for_ndim(ndim));

    auto d_A = dm.create_ro_buffer(sizeof(float) * size_A, A);
    auto d_B = dm.create_ro_buffer(sizeof(float) * size_B, B);
    auto d_C = dm.create_rw_buffer(sizeof(float) * size_C, C);

    auto d_strides_A = dm.create_ro_buffer(sizeof(long) * ndim, strides_A);
    auto d_strides_B = dm.create_ro_buffer(sizeof(long) * ndim, strides_B);
    auto d_strides_C = dm.create_ro_buffer(sizeof(long) * ndim, strides_C);
    auto d_shape     = dm.create_ro_buffer(sizeof(long) * ndim, shape);

    kernel.setArg(0,  d_A);
    kernel.setArg(1,  d_strides_A);
    kernel.setArg(2,  d_B);
    kernel.setArg(3,  d_strides_B);
    kernel.setArg(4,  d_C);
    kernel.setArg(5,  d_strides_C);
    kernel.setArg(6,  d_shape);
    kernel.setArg(7,  ndim);
    kernel.setArg(8,  total_elements);

    dm.launch(kernel, cl::NDRange(total_elements), cl::NullRange, "add_v1");
    dm.read_buffer(d_C, sizeof(float) * size_C, C);
}

// ---------------------------------------------------------------------------
// std::vector convenience
// ---------------------------------------------------------------------------
void launch_add_v1(const std::vector<float>& A, const std::vector<long>& strides_A,
                   const std::vector<float>& B, const std::vector<long>& strides_B,
                         std::vector<float>& C, const std::vector<long>& strides_C,
                   const std::vector<long>& shape) {

    long total = 1;
    for (long s : shape) total *= s;

    launch_add_v1(A.data(), strides_A.data(), (long)A.size(),
                  B.data(), strides_B.data(), (long)B.size(),
                  C.data(), strides_C.data(), (long)C.size(),
                  shape.data(), (int)shape.size(), total);
}

// ---------------------------------------------------------------------------
// v0 overloads (kept for reference)
// ---------------------------------------------------------------------------
void launch_add_v0(const std::vector<float>& A,    const std::vector<long>& strides_A,
                   const std::vector<float>& B,    const std::vector<long>& strides_B,
                         std::vector<float>& C,    const std::vector<long>& strides_C,
                   const std::vector<long>& shape) {

    long total = 1;
    for (long s : shape) total *= s;

    launch_add_v0(A.data(), strides_A.data(), (long)A.size(),
                  B.data(), strides_B.data(), (long)B.size(),
                  C.data(), strides_C.data(), (long)C.size(),
                  shape.data(), (int)shape.size(), total);
}
