#include "flash_attn.h"

// ===========================================================================
// flash_attn_v0 - your first version (currently a TODO placeholder, not
// registered yet).
//
// Suggested goal: the most straightforward O(Nq*Nk*d) implementation, just to
// get the test / benchmark pipeline working.  Then iterate with blocking,
// online softmax, SIMD and OpenMP.
//
// Implementation hints (not mandatory):
//   - index with params.q_stride_b/h/s etc.; do not assume the tensors are
//     contiguous;
//   - per query row: max -> exp/sum -> weighted sum of V; with causal, skip
//     j > i;
//   - locate the output with params.o_stride_*; it must be fully written
//     (including masked rows);
//   - use params.softmax_scale; do not modify q/k/v.
//
// When done, go to flash_attn_versions.cpp and:
//   1. uncomment the `void flash_attn_v0(...)` declaration;
//   2. add {"v0", flash_attn_v0} to flash_attn_versions().
//
// Then:
//   cmake --build build --target flash_attn_test flash_attn_benchmark -j
//   cd build && ./src/cpu/flash_attn/flash_attn_test
//   cd build && ./src/cpu/flash_attn/flash_attn_benchmark
// ===========================================================================
void flash_attn_v0(const FlashAttentionParams& params) {
  flash_attn_check_params(params);
  // TODO(user): implement v0 here.
  (void) params;
}
