#pragma once

#include <cstdint>
#include <vector>

// ===========================================================================
// CPU FlashAttention - fp32, layout [B, H, N, d] (row-major)
//
//   S = scale * Q @ K^T                    [Nq, Nk]
//   P = softmax(S)                         causal: elements with j > i are
//   dropped O = P @ V                              [Nq, d]
//
// Data layout (all fp32, row-major):
//   q : [B, H, Nq, d]
//   k : [B, H, Nk, d]
//   v : [B, H, Nk, d]
//   o : [B, H, Nq, d]
//
// Strides are in ELEMENTS (not bytes) and cover the batch / head / row dims.
// The innermost head_dim dim is assumed contiguous (stride 1 within a row).
//
// A version is a plain function `void version(const FlashAttentionParams&)`:
// it reads q/k/v, fully overwrites o (including rows where every key is masked;
// writing zeros is fine) and must not modify q/k/v.  Register it in
// flash_attn_versions.cpp and both test.cpp and benchmark.cpp pick it up.
// ===========================================================================
struct FlashAttentionParams {
  const float* q = nullptr;
  const float* k = nullptr;
  const float* v = nullptr;
  float* o = nullptr;

  int batch = 0;
  int num_heads = 0;
  int seq_len_q = 0;
  int seq_len_k = 0;
  int head_dim = 0;

  // Strides in elements.
  int64_t q_stride_b = 0;
  int64_t q_stride_h = 0;
  int64_t q_stride_s = 0;
  int64_t k_stride_b = 0;
  int64_t k_stride_h = 0;
  int64_t k_stride_s = 0;
  int64_t v_stride_b = 0;
  int64_t v_stride_h = 0;
  int64_t v_stride_s = 0;
  int64_t o_stride_b = 0;
  int64_t o_stride_h = 0;
  int64_t o_stride_s = 0;

  float softmax_scale = 0.0f;  // usually 1 / sqrt(head_dim)
  bool causal = false;
};

// ---------------------------------------------------------------------------
// Host helpers for contiguous row-major tensors.
// ---------------------------------------------------------------------------
// scale = 1 / sqrt(head_dim)
FlashAttentionParams make_flash_attn_params(const float* q, const float* k,
                                            const float* v, float* o, int batch,
                                            int num_heads, int seq_len_q,
                                            int seq_len_k, int head_dim,
                                            bool causal);

// Explicit softmax scale.
FlashAttentionParams make_flash_attn_params(const float* q, const float* k,
                                            const float* v, float* o, int batch,
                                            int num_heads, int seq_len_q,
                                            int seq_len_k, int head_dim,
                                            float softmax_scale, bool causal);

// Validate pointers, shapes, strides and scale.  Throws std::invalid_argument
// on misuse.  Every version may call this once at its entry point.
void flash_attn_check_params(const FlashAttentionParams& p);

// ---------------------------------------------------------------------------
// Version registry (single place shared by test.cpp and benchmark.cpp)
// ---------------------------------------------------------------------------
using FlashAttnFn = void (*)(const FlashAttentionParams&);

struct FlashAttnVersion {
  const char* name;
  FlashAttnFn fn;
};

// Defined in flash_attn_versions.cpp.  Empty until the first version is
// registered -- test cases are skipped and the benchmark prints a hint.
std::vector<FlashAttnVersion> flash_attn_versions();
