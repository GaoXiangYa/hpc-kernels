add_library(hgemm_cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/hgemm.cu)
target_include_directories(hgemm_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(hgemm_cuda PUBLIC CUDA::cudart)

if(USE_TORCH)
  add_executable(hgemm_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
  target_link_libraries(hgemm_test PRIVATE hgemm_cuda GTest::gtest_main
    /opt/libtorch/lib/libtorch_cpu.so
    /opt/libtorch/lib/libc10.so
  )
  target_include_directories(hgemm_test PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

add_executable(hgemm_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cu)
target_link_libraries(hgemm_benchmark PRIVATE hgemm_cuda CUDA::cublas)
