#pragma once
#include <cstddef>
#include <vector>

// v0 — generic loop (works for any ndim, baseline)
void launch_add_v0(const float* A, const long* strides_A, long size_A,
                   const float* B, const long* strides_B, long size_B,
                         float* C, const long* strides_C, long size_C,
                   const long* shape, int ndim, long total_elements);

void launch_add_v0(const std::vector<float>& A, const std::vector<long>& strides_A,
                   const std::vector<float>& B, const std::vector<long>& strides_B,
                         std::vector<float>& C, const std::vector<long>& strides_C,
                   const std::vector<long>& shape);

// v1 — ndim-specialized: dim1/2/3 hand-unrolled, dim4+ fallback
void launch_add_v1(const float* A, const long* strides_A, long size_A,
                   const float* B, const long* strides_B, long size_B,
                         float* C, const long* strides_C, long size_C,
                   const long* shape, int ndim, long total_elements);

void launch_add_v1(const std::vector<float>& A, const std::vector<long>& strides_A,
                   const std::vector<float>& B, const std::vector<long>& strides_B,
                         std::vector<float>& C, const std::vector<long>& strides_C,
                   const std::vector<long>& shape);
