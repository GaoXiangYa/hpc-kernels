include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)

# Reusable helper for kernel subdirectories
function(add_ocl_kernel NAME)
    add_library(${NAME}_ocl SHARED ${NAME}.cpp)
    target_link_libraries(${NAME}_ocl PRIVATE OpenCL::OpenCL ocl_common)

    add_executable(${NAME}_test test.cpp)
    target_link_libraries(${NAME}_test PRIVATE ${NAME}_ocl ocl_common
                                              OpenCL::OpenCL
                                              GTest::GTest GTest::Main)
endfunction()

add_subdirectory(common)
add_subdirectory(reduce)
add_subdirectory(gemm)
add_subdirectory(conv2d)
add_subdirectory(softmax)
