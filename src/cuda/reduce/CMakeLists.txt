add_library(reduce_cuda 
            STATIC 
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_v0.cu 
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_v1.cu 
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_v2.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_v3.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_v4.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_v5.cu)

target_include_directories(reduce_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

add_executable(reduce_test ${CMAKE_CURRENT_SOURCE_DIR}/reduce_test.cpp)
target_link_libraries(reduce_test PRIVATE reduce_cuda GTest::gtest_main)

add_executable(reduce_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/reduce_benchmark.cpp)
target_link_libraries(reduce_benchmark PRIVATE reduce_cuda GTest::gtest_main)
