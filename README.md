# HPC-KERNELS

Practice writing high-performance compute kernels for GPU (CUDA, OpenCL) and CPU (x86 SIMD).

## Project Structure

```
hpc-kernels/
├── CMakeLists.txt          # Top-level build (option-gated backends)
├── cmake/                  # CMake modules for external dependencies
├── include/                # Shared utility headers
├── scripts/                # Python plotting/analysis scripts
├── docs/                   # Documentation and assets
├── third_party/            # Vendored third-party libraries
└── src/
    ├── cuda/
    │   ├── common/         # CUDA utilities (benchmark, init_random, CHECK_CUDA)
    │   ├── gemm/           # Matrix multiplication (v0-v3)
    │   ├── gemv/           # Matrix-vector product (v0-v6, cutlass, cublas)
    │   ├── reduce/         # Reduction (v0-v5)
    │   ├── rmsnorm/        # RMS normalization (v0-v3, flashinfer)
    │   └── softmax/        # Softmax (v0-v1)
    ├── opencl/
    │   ├── common/         # OpenCL utilities (OCLKernel, profiling)
    │   ├── gemm/           # Matrix multiplication (v0-v10)
    │   ├── reduce/         # Reduction (v0-v4)
    │   ├── conv2d/         # 2D convolution (v0-v4)
    │   └── softmax/        # Softmax with attention mask (v0)
    └── cpu/
        ├── common/         # CPU utilities (initMatrix, printMatrix)
        └── gemm/           # Matrix multiplication (v0-v20, openmp, blas)
```

## Backend Support

Use CMake options to enable backends:

```bash
cmake -DUSE_CUDA=ON -DUSE_OPENCL=ON -DUSE_CPU=ON ..
```

For FlashInfer and CUTLASS support, set environment variables:
```bash
export FLASHINFER_PATH=/path/to/flashinfer
export CUTLASS_PATH=/path/to/cutlass
```

## CUDA

- [x] gemm (v0-v3)
- [x] gemv (v0-v6, cutlass, cublas)
- [x] reduce (v0-v5)
- [x] rmsnorm (v0-v3, flashinfer)
- [x] softmax (v0-v1)

## OpenCL

- [x] gemm (v0-v10)
- [x] reduce (v0-v4)
- [x] conv2d (v0-v4)
- [x] softmax (v0)

## CPU

- [x] gemm (v0-v20, openmp, blas)
