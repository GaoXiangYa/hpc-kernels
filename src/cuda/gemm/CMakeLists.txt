add_library(gemm_cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v0.cu)
target_include_directories(gemm_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

if(USE_TORCH)
  add_executable(gemm_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
  target_link_libraries(gemm_test PRIVATE gemm_cuda GTest::gtest_main
    /opt/libtorch/lib/libtorch_cpu.so
    /opt/libtorch/lib/libc10.so
  )
  target_include_directories(gemm_test PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

add_executable(gemm_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cu)
target_link_libraries(gemm_benchmark PRIVATE gemm_cuda)
