#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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

template <int BlockKV>
struct SoftmaxState {
  float m = -INFINITY;
  float l = 0.0f;
  float acc[BlockKV];

  void reset() {
    m = -INFINITY;
    l = 0.0f;
    std::memset(acc, 0, BlockKV * sizeof(float));
  }
};

static constexpr int kBlockSeqQ = 64;
static constexpr int kBlockSeqKV = 64;
static constexpr int kBlockHeadDims = 64;

static inline int64_t q_offset(const FlashAttentionParams& params, int b, int h,
                               int s) {
  return b * params.q_stride_b + h * params.q_stride_h + s * params.q_stride_s;
}

static inline int64_t o_offset(const FlashAttentionParams& params, int b, int h,
                               int s) {
  return b * params.o_stride_b + h * params.o_stride_h + s * params.o_stride_s;
}

static inline int64_t k_offset(const FlashAttentionParams& params, int b,
                                int h, int s) {
  return b * params.k_stride_b + h * params.k_stride_h + s * params.k_stride_s;
}

static inline int64_t v_offset(const FlashAttentionParams& params, int b,
                                int h, int s) {
  return b * params.v_stride_b + h * params.v_stride_h + s * params.v_stride_s;
}

template <int BlockM, int BlockHeadDim>
static inline void load_q_block(const float* q_start, float* block_q,
                                int q_stride_s, int block_size) {
  for (int i = 0; i < block_size; ++i) {
    const float* src = q_start + i * q_stride_s;

    float* dst = block_q + i * BlockHeadDim;

    std::copy_n(src, BlockHeadDim, dst);
  }
}

template <int BlockN, int BlockHeadDim>
static inline void load_kv_block(const float* kv_start, float* block_kv,
                                 int kv_stride_s, int block_size) {
  for (int i = 0; i < block_size; ++i) {
    const float* src = kv_start + i * kv_stride_s;

    float* dst = block_kv + i * BlockHeadDim;

    std::copy_n(src, BlockHeadDim, dst);
  }
}

template <int BlockSeqQ, int BlockHeadDim>
static inline void store_o_block(float* o_start, const float* block_o,
                                 int o_stride_s, int block_size) {
  for (int i = 0; i < block_size; ++i) {
    float* dst = o_start + i * o_stride_s;

    const float* src = block_o + i * BlockHeadDim;

    std::copy_n(src, BlockHeadDim, dst);
  }
}

template <int BlockM, int BlockN, int BlockHeadDim>
static inline void matmul_qk(const float* q, const float* k, float* p) {
  for (int im = 0; im < BlockM; ++im) {
    for (int in = 0; in < BlockN; ++in) {
      float sum = 0.0f;
      for (int ik = 0; ik < BlockHeadDim; ++ik) {
        sum += q[im * BlockHeadDim + ik] * k[in * BlockHeadDim + ik];
      }
      p[im * BlockN + in] = sum;
    }
  }
}

template <int BlockM, int BlockN, int BlockHeadDim>
static inline void matmul_pv_online_softmax(const float* p, const float* v,
                                            SoftmaxState<BlockHeadDim>* state) {
  for (int m = 0; m < BlockM; ++m) {
    // online softmax
    float m_new = -INFINITY;
    for (int n = 0; n < BlockN; ++n) {
      m_new = std::max(m_new, p[m * BlockN + n]);
    }

    float alpha = std::exp(state[m].m - m_new);
    float sum_l = 0.0f;

    for (int n = 0; n < BlockN; ++n) {
      float val = std::exp(p[m * BlockN + n] - m_new);
      sum_l += val;
    }

    state[m].l = alpha * state[m].l + sum_l;
    state[m].m = m_new;

    // matmul p * v
    float sum_matmul = 0.0f;
    for (int n = 0; n < BlockN; ++n) {
      float val = std::exp(p[m * BlockN + n] - m_new);
      for (int d = 0; d < BlockHeadDim; ++d) {
        state[m].acc[d] =
            alpha * state[m].acc[d] + val * v[n * BlockHeadDim + d];
      }
    }
  }
}

template <int BlockSeqQ, int BlockHeadDims>
static inline void
normailize_output(std::array<float, BlockSeqQ * BlockHeadDims>& output,
                  std::array<SoftmaxState<BlockHeadDims>, BlockSeqQ>& state) {
  for (int q = 0; q < BlockSeqQ; ++q) {
    for (int d = 0; d < BlockHeadDims; ++d) {
      output[q * BlockHeadDims + d] = state[q].acc[d] / state[q].l;
    }
  }
}

void flash_attn_v0(const FlashAttentionParams& params) {
  flash_attn_check_params(params);

  std::array<float, kBlockHeadDims * kBlockSeqQ> block_q{};
  std::array<float, kBlockSeqQ * kBlockHeadDims> block_o{};
  std::array<float, kBlockHeadDims * kBlockSeqQ> block_k{};
  std::array<float, kBlockHeadDims * kBlockSeqQ> block_v{};
  std::array<SoftmaxState<kBlockHeadDims>, kBlockSeqQ> state{};
  std::array<float, kBlockSeqQ * kBlockSeqKV> block_p{};

  // TODO(user): implement v0 here.
  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < params.num_heads; ++h) {
      for (int q_start = 0; q_start < params.seq_len_q; q_start += kBlockSeqQ) {
        const int q_block_size =
            std::min(kBlockSeqQ, params.seq_len_q - q_start);

        const auto q_off = q_offset(params, b, h, q_start);

        load_q_block<kBlockSeqQ, kBlockHeadDims>(params.q + q_off,
                                                 block_q.data(),
                                                 params.q_stride_s,
                                                 q_block_size);

        //  init online softmax state
        for (auto& st : state) {
          st.reset();
        }

        for (int kv_start = 0; kv_start < params.seq_len_k;
             kv_start += kBlockSeqKV) {
          const int kv_block_size =
              std::min(kBlockSeqKV, params.seq_len_k - kv_start);

          const auto k_off = k_offset(params, b, h, kv_start);
          const auto v_off = v_offset(params, b, h, kv_start);

          load_kv_block<kBlockSeqKV, kBlockHeadDims>(params.k + k_off,
                                                     block_k.data(),
                                                     params.k_stride_s,
                                                     kv_block_size);

          load_kv_block<kBlockSeqKV, kBlockHeadDims>(params.v + v_off,
                                                     block_v.data(),
                                                     params.v_stride_s,
                                                     kv_block_size);

          matmul_qk<kBlockSeqQ, kBlockSeqKV, kBlockHeadDims>(block_q.data(),
                                                             block_k.data(),
                                                             block_p.data());

          matmul_pv_online_softmax<kBlockSeqQ, kBlockSeqKV, kBlockHeadDims>(
              block_p.data(), block_v.data(), state.data());
        }

        normailize_output<kBlockSeqQ, kBlockHeadDims>(block_o, state);

        const auto o_off = o_offset(params, b, h, q_start);
        const int o_block_size =
            std::min(kBlockSeqQ, params.seq_len_q - q_start);
        store_o_block<kBlockSeqQ, kBlockHeadDims>(params.o + o_off,
                                                  block_o.data(),
                                                  params.o_stride_s,
                                                  o_block_size);
      }
    }
  }
}
