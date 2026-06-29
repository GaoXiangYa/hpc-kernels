add_library(gemm_cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cu)
target_include_directories(gemm_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

add_executable(gemm_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(gemm_test PRIVATE gemm_cuda GTest::gtest_main)
