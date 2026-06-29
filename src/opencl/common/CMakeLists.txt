add_library(ocl_common STATIC device.cpp)
target_link_libraries(ocl_common PUBLIC OpenCL::OpenCL)
target_include_directories(ocl_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ocl_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
