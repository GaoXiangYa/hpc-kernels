include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# gemm needs BLAS.  Skip it when BLAS is unavailable so CPU operators that do
# not need BLAS (flash_attn, ...) can still be built; install BLAS and
# reconfigure if you want gemm.
find_package(BLAS QUIET)
if(BLAS_FOUND)
  add_subdirectory(gemm)
else()
  message(STATUS "CPU gemm skipped: BLAS not found")
endif()

add_subdirectory(flash_attn)
