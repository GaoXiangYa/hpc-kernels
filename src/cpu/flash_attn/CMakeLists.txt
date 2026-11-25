find_package(OpenMP REQUIRED)

add_library(flash_attn_cpu STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn_versions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn_v0.cpp
  # add flash_attn_vN.cpp here when adding a new version
)
target_include_directories(flash_attn_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(flash_attn_cpu PUBLIC OpenMP::OpenMP_CXX)

add_executable(flash_attn_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(flash_attn_test PRIVATE flash_attn_cpu GTest::GTest
                                               GTest::Main)

# Optional: use libtorch only as an independent oracle for the reference
# implementation (the kernels themselves stay dependency-free).
if(USE_TORCH)
  list(APPEND CMAKE_PREFIX_PATH /opt/libtorch)
  find_package(Torch REQUIRED)
  target_compile_definitions(flash_attn_test PRIVATE FLASH_ATTN_USE_TORCH=1)
  target_include_directories(flash_attn_test PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(flash_attn_test PRIVATE torch_cpu c10)
endif()

add_executable(flash_attn_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp)
target_link_libraries(flash_attn_benchmark PRIVATE flash_attn_cpu)
