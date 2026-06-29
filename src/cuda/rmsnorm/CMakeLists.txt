add_library(rmsnorm_cuda 
            STATIC 
            ${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_rmsnorm.cu 
            ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm_v0.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm_v1.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm_v2.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm_v3.cu
            )

target_include_directories(rmsnorm_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

add_executable(rmsnorm_test ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm_test.cpp)
target_link_libraries(rmsnorm_test PRIVATE rmsnorm_cuda GTest::gtest_main)

add_executable(rmsnorm_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm_benchmark.cpp)
target_link_libraries(rmsnorm_benchmark PRIVATE rmsnorm_cuda GTest::gtest_main)
