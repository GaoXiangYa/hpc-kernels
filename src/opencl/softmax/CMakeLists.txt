list(APPEND CMAKE_PREFIX_PATH /opt/libtorch)
find_package(Torch REQUIRED)
add_ocl_kernel(softmax ${TORCH_LIBRARIES})
target_include_directories(softmax_test PRIVATE ${TORCH_INCLUDE_DIRS})
