add_library(gemv_cuda STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_v0.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_v1.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_v2.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_v3.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_v4.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_v5.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_v6.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_cutlass.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv_cublas.cu
)
target_include_directories(gemv_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

add_executable(gemv_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(gemv_test PRIVATE CUDA::cusparse CUDA::cublas gemv_cuda GTest::gtest_main)

add_executable(gemv_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp)
target_link_libraries(gemv_benchmark PRIVATE CUDA::cublas gemv_cuda)
