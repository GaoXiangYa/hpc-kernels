#include "flash_attn.h"

// ===========================================================================
// Version registry
//
// Adding a version only requires two edits in this file:
//   1. declare the version function below (implemented in flash_attn_vN.cpp);
//   2. add { "vN", flash_attn_vN } to the list returned by
//      flash_attn_versions().
//
// test.cpp and benchmark.cpp both read the registry, so they never need to be
// touched.  Unregistered versions are neither tested nor benchmarked.
// ===========================================================================
void flash_attn_v0(const FlashAttentionParams&);

std::vector<FlashAttnVersion> flash_attn_versions() {
  return {
      {"v0", flash_attn_v0},
  };
}
