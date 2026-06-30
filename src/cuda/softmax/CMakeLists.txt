add_library(softmax_cuda STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/softmax_v0.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/softmax_v1.cu
)
target_include_directories(softmax_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

add_executable(softmax_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(softmax_test PRIVATE softmax_cuda GTest::gtest_main)

