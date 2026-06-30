cmake_minimum_required(VERSION 3.20)

project(hpc-kernels LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTest REQUIRED)

option(USE_CPU "use cpu backend" OFF)
option(USE_CUDA "use cuda backend" OFF)
option(USE_OPENCL "use opencl backend" OFF)
option(USE_TORCH "use libtorch for testing" ON)

if(USE_TORCH)
  find_package(Torch REQUIRED PATHS /opt/libtorch)
endif()

if(USE_CPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -mavx -mavx2 -O3")
  add_subdirectory(src/cpu)
endif()

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  add_subdirectory(src/cuda)
endif()

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  add_subdirectory(src/opencl)
endif()
