#include "flash_attn.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void check_positive(int value, const char* what) {
  if (value <= 0) {
    throw std::invalid_argument(std::string("flash_attn: ") + what +
                                " must be positive");
  }
}

void check_pointer(const void* ptr, const char* what) {
  if (ptr == nullptr) {
    throw std::invalid_argument(std::string("flash_attn: ") + what +
                                " must not be null");
  }
}

// A tensor is [batch, num_heads, seq_len, head_dim] with the innermost dim
// contiguous.  Larger strides are allowed (padding), overlapping ones are not.
void check_strides(const FlashAttentionParams& p, const char* name,
                   int64_t stride_b, int64_t stride_h, int64_t stride_s,
                   int seq_len) {
  const int64_t row_extent =
      static_cast<int64_t>(seq_len - 1) * stride_s + p.head_dim;
  if (stride_s < p.head_dim) {
    throw std::invalid_argument(std::string("flash_attn: ") + name +
                                "_stride_s is smaller than head_dim");
  }
  if (stride_h < row_extent) {
    throw std::invalid_argument(std::string("flash_attn: ") + name +
                                "_stride_h is smaller than a sequence row");
  }
  if (stride_b <
      static_cast<int64_t>(p.num_heads - 1) * stride_h + row_extent) {
    throw std::invalid_argument(std::string("flash_attn: ") + name +
                                "_stride_b is smaller than one head block");
  }
}

FlashAttentionParams make_contiguous_params(const float* q, const float* k,
                                            const float* v, float* o, int batch,
                                            int num_heads, int seq_len_q,
                                            int seq_len_k, int head_dim,
                                            float softmax_scale, bool causal) {
  FlashAttentionParams p;
  p.q = q;
  p.k = k;
  p.v = v;
  p.o = o;
  p.batch = batch;
  p.num_heads = num_heads;
  p.seq_len_q = seq_len_q;
  p.seq_len_k = seq_len_k;
  p.head_dim = head_dim;

  const int64_t hd = head_dim;
  p.q_stride_s = hd;
  p.q_stride_h = static_cast<int64_t>(seq_len_q) * hd;
  p.q_stride_b = static_cast<int64_t>(num_heads) * p.q_stride_h;
  p.k_stride_s = hd;
  p.k_stride_h = static_cast<int64_t>(seq_len_k) * hd;
  p.k_stride_b = static_cast<int64_t>(num_heads) * p.k_stride_h;
  p.v_stride_s = hd;
  p.v_stride_h = static_cast<int64_t>(seq_len_k) * hd;
  p.v_stride_b = static_cast<int64_t>(num_heads) * p.v_stride_h;
  p.o_stride_s = hd;
  p.o_stride_h = static_cast<int64_t>(seq_len_q) * hd;
  p.o_stride_b = static_cast<int64_t>(num_heads) * p.o_stride_h;

  p.softmax_scale = softmax_scale;
  p.causal = causal;
  return p;
}

}  // namespace

FlashAttentionParams make_flash_attn_params(const float* q, const float* k,
                                            const float* v, float* o, int batch,
                                            int num_heads, int seq_len_q,
                                            int seq_len_k, int head_dim,
                                            bool causal) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  return make_contiguous_params(q, k, v, o, batch, num_heads, seq_len_q,
                                seq_len_k, head_dim, scale, causal);
}

FlashAttentionParams make_flash_attn_params(const float* q, const float* k,
                                            const float* v, float* o, int batch,
                                            int num_heads, int seq_len_q,
                                            int seq_len_k, int head_dim,
                                            float softmax_scale, bool causal) {
  return make_contiguous_params(q, k, v, o, batch, num_heads, seq_len_q,
                                seq_len_k, head_dim, softmax_scale, causal);
}

void flash_attn_check_params(const FlashAttentionParams& p) {
  check_positive(p.batch, "batch");
  check_positive(p.num_heads, "num_heads");
  check_positive(p.seq_len_q, "seq_len_q");
  check_positive(p.seq_len_k, "seq_len_k");
  check_positive(p.head_dim, "head_dim");
  check_pointer(p.q, "q");
  check_pointer(p.k, "k");
  check_pointer(p.v, "v");
  check_pointer(p.o, "o");

  if (!std::isfinite(p.softmax_scale) || p.softmax_scale <= 0.0f) {
    throw std::invalid_argument(
        "flash_attn: softmax_scale must be finite and positive");
  }

  check_strides(p, "q", p.q_stride_b, p.q_stride_h, p.q_stride_s, p.seq_len_q);
  check_strides(p, "k", p.k_stride_b, p.k_stride_h, p.k_stride_s, p.seq_len_k);
  check_strides(p, "v", p.v_stride_b, p.v_stride_h, p.v_stride_s, p.seq_len_k);
  check_strides(p, "o", p.o_stride_b, p.o_stride_h, p.o_stride_s, p.seq_len_q);
}
