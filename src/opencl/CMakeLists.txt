include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(reduce)
add_subdirectory(gemm)
add_subdirectory(conv2d)
add_subdirectory(softmax)