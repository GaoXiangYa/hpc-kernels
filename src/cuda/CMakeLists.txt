include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# External dependencies - set these paths or use CMake find modules
if(DEFINED ENV{FLASHINFER_PATH})
  include_directories($ENV{FLASHINFER_PATH}/include)
endif()
if(DEFINED ENV{CUTLASS_PATH})
  include_directories($ENV{CUTLASS_PATH}/include)
endif()

add_subdirectory(gemm)
add_subdirectory(reduce)
add_subdirectory(rmsnorm)
add_subdirectory(gemv)
add_subdirectory(softmax)
