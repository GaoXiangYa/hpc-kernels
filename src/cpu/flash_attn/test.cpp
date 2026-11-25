#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef FLASH_ATTN_USE_TORCH
#include <torch/torch.h>
#endif

#include "flash_attn.h"

namespace {

constexpr const char* kNoVersionsHint =
    "no versions registered: implement src/cpu/flash_attn/flash_attn_v0.cpp "
    "and register it in flash_attn_versions.cpp";

std::vector<float> random_vec(std::size_t n, std::uint32_t seed,
                              float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> values(n);
  for (auto& x : values)
    x = dist(gen);
  return values;
}

// ---------------------------------------------------------------------------
// Naive double-precision oracle: O(Nq * Nk * d), no blocking, no online
// softmax.  Slow but obviously correct; independent of the kernel versions.
// ---------------------------------------------------------------------------
void reference_attention(const FlashAttentionParams& p) {
  const double scale = p.softmax_scale;
  const double neg_inf = -std::numeric_limits<double>::infinity();
  std::vector<double> scores(p.seq_len_k);

  for (int b = 0; b < p.batch; ++b) {
    for (int h = 0; h < p.num_heads; ++h) {
      const float* q_bh = p.q + static_cast<std::int64_t>(b) * p.q_stride_b +
                          static_cast<std::int64_t>(h) * p.q_stride_h;
      const float* k_bh = p.k + static_cast<std::int64_t>(b) * p.k_stride_b +
                          static_cast<std::int64_t>(h) * p.k_stride_h;
      const float* v_bh = p.v + static_cast<std::int64_t>(b) * p.v_stride_b +
                          static_cast<std::int64_t>(h) * p.v_stride_h;

      for (int i = 0; i < p.seq_len_q; ++i) {
        const float* q_row = q_bh + static_cast<std::int64_t>(i) * p.q_stride_s;
        double row_max = neg_inf;
        for (int j = 0; j < p.seq_len_k; ++j) {
          if (p.causal && j > i) {
            scores[j] = neg_inf;
            continue;
          }
          const float* k_row =
              k_bh + static_cast<std::int64_t>(j) * p.k_stride_s;
          double acc = 0.0;
          for (int dd = 0; dd < p.head_dim; ++dd) {
            acc += static_cast<double>(q_row[dd]) * k_row[dd];
          }
          scores[j] = acc * scale;
          row_max = std::max(row_max, scores[j]);
        }

        double denom = 0.0;
        for (int j = 0; j < p.seq_len_k; ++j) {
          if (scores[j] == neg_inf) {
            scores[j] = 0.0;
            continue;
          }
          scores[j] = std::exp(scores[j] - row_max);
          denom += scores[j];
        }

        float* o_row = p.o + static_cast<std::int64_t>(b) * p.o_stride_b +
                       static_cast<std::int64_t>(h) * p.o_stride_h +
                       static_cast<std::int64_t>(i) * p.o_stride_s;
        for (int dd = 0; dd < p.head_dim; ++dd) {
          double acc = 0.0;
          for (int j = 0; j < p.seq_len_k; ++j) {
            const float* v_row =
                v_bh + static_cast<std::int64_t>(j) * p.v_stride_s;
            acc += scores[j] * static_cast<double>(v_row[dd]);
          }
          o_row[dd] = static_cast<float>(acc / denom);
        }
      }
    }
  }
}

#ifdef FLASH_ATTN_USE_TORCH
std::vector<float> torch_attention(const std::vector<float>& q,
                                   const std::vector<float>& k,
                                   const std::vector<float>& v, int batch,
                                   int heads, int seq_q, int seq_k, int d,
                                   float scale, bool causal) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto tq = torch::from_blob(const_cast<float*>(q.data()),
                             {batch, heads, seq_q, d}, opts)
                .clone();
  auto tk = torch::from_blob(const_cast<float*>(k.data()),
                             {batch, heads, seq_k, d}, opts)
                .clone();
  auto tv = torch::from_blob(const_cast<float*>(v.data()),
                             {batch, heads, seq_k, d}, opts)
                .clone();

  auto scores = torch::matmul(tq, tk.transpose(-1, -2)) * scale;
  if (causal) {
    auto idx_q = torch::arange(seq_q, opts).unsqueeze(1);  // [Nq, 1]
    auto idx_k = torch::arange(seq_k, opts).unsqueeze(0);  // [1, Nk]
    auto mask = idx_k > idx_q;                             // j > i
    scores = scores.masked_fill(mask, -std::numeric_limits<float>::infinity());
  }
  auto out = torch::matmul(torch::softmax(scores, -1), tv).contiguous();
  const auto* ptr = out.data_ptr<float>();
  return std::vector<float>(ptr, ptr + out.numel());
}
#endif

// ---------------------------------------------------------------------------
// Test cases: [B, H, Nq, Nk, d], causal, absolute tolerance.
// d intentionally includes 1, 48 and 128 (non-vector-multiple / large head).
// ---------------------------------------------------------------------------
struct Case {
  int batch;
  int heads;
  int seq_q;
  int seq_k;
  int head_dim;
  bool causal;
  float atol;
};

class FlashAttnTest : public ::testing::Test {
protected:
  void run_case(const Case& c) {
    const auto versions = flash_attn_versions();
    if (versions.empty()) {
      GTEST_SKIP() << kNoVersionsHint;
    }

    const std::size_t q_elems =
        static_cast<std::size_t>(c.batch) * c.heads * c.seq_q * c.head_dim;
    const std::size_t kv_elems =
        static_cast<std::size_t>(c.batch) * c.heads * c.seq_k * c.head_dim;

    auto q = random_vec(q_elems, 1);
    auto k = random_vec(kv_elems, 2);
    auto v = random_vec(kv_elems, 3);

    std::vector<float> o_ref(q_elems, 0.0f);
    auto p_ref = make_flash_attn_params(q.data(), k.data(), v.data(),
                                        o_ref.data(), c.batch, c.heads, c.seq_q,
                                        c.seq_k, c.head_dim, c.causal);
    reference_attention(p_ref);

    for (const auto& version : versions) {
      SCOPED_TRACE(version.name);
      std::vector<float> o(q_elems,
                           -12345.0f);  // sentinel: output must be written
      auto p = make_flash_attn_params(q.data(), k.data(), v.data(), o.data(),
                                      c.batch, c.heads, c.seq_q, c.seq_k,
                                      c.head_dim, c.causal);
      version.fn(p);
      for (std::size_t i = 0; i < q_elems; ++i) {
        EXPECT_NEAR(o_ref[i], o[i], c.atol) << "at flat index " << i;
      }
    }
  }
};

#define FLASH_ATTN_TEST_CASE(name, B, H, NQ, NK, D, CAUSAL, ATOL) \
  TEST_F(FlashAttnTest, name) {                                   \
    run_case({B, H, NQ, NK, D, CAUSAL, ATOL});                    \
  }

FLASH_ATTN_TEST_CASE(one_token, 1, 1, 1, 1, 1, false, 1e-4f)
FLASH_ATTN_TEST_CASE(small, 1, 1, 4, 4, 8, false, 1e-4f)
FLASH_ATTN_TEST_CASE(multi_head, 1, 2, 8, 8, 16, false, 1e-4f)
FLASH_ATTN_TEST_CASE(causal_small, 1, 1, 16, 16, 32, true, 1e-4f)
FLASH_ATTN_TEST_CASE(d64, 1, 4, 64, 64, 64, false, 1e-4f)
FLASH_ATTN_TEST_CASE(d64_causal, 1, 2, 128, 128, 64, true, 1e-4f)
FLASH_ATTN_TEST_CASE(batched, 2, 4, 128, 128, 64, false, 1e-4f)
FLASH_ATTN_TEST_CASE(odd_100_d48, 1, 2, 100, 100, 48, true, 1e-4f)
FLASH_ATTN_TEST_CASE(cross_attention, 1, 2, 32, 64, 64, false, 1e-4f)
FLASH_ATTN_TEST_CASE(causal_short_k, 1, 2, 64, 32, 64, true, 1e-4f)
FLASH_ATTN_TEST_CASE(long_causal, 1, 1, 512, 512, 64, true, 5e-4f)
FLASH_ATTN_TEST_CASE(long, 1, 1, 512, 512, 64, false, 5e-4f)

// ---------------------------------------------------------------------------
// Numerical stability: large scores make exp() overflow without max
// subtraction.  The oracle is double, so this only tests the kernels.
// ---------------------------------------------------------------------------
TEST_F(FlashAttnTest, numerical_stability) {
  const auto versions = flash_attn_versions();
  if (versions.empty()) {
    GTEST_SKIP() << kNoVersionsHint;
  }

  const int b = 1, h = 1, nq = 64, nk = 64, d = 64;
  const std::size_t q_elems = static_cast<std::size_t>(nq) * d;
  const std::size_t kv_elems = static_cast<std::size_t>(nk) * d;
  auto q = random_vec(q_elems, 11, -100.0f, 100.0f);
  auto k = random_vec(kv_elems, 12, -100.0f, 100.0f);
  auto v = random_vec(kv_elems, 13, -1.0f, 1.0f);

  std::vector<float> o_ref(q_elems, 0.0f);
  auto p_ref =
      make_flash_attn_params(q.data(), k.data(), v.data(), o_ref.data(), b, h,
                             nq, nk, d, /*causal=*/false);
  reference_attention(p_ref);

  for (const auto& version : versions) {
    SCOPED_TRACE(version.name);
    std::vector<float> o(q_elems, -12345.0f);
    auto p = make_flash_attn_params(q.data(), k.data(), v.data(), o.data(), b,
                                    h, nq, nk, d, /*causal=*/false);
    version.fn(p);
    for (std::size_t i = 0; i < q_elems; ++i) {
      EXPECT_NEAR(o_ref[i], o[i], 1e-4f) << "at flat index " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// Framework / reference self-checks (run even when no version is registered)
// ---------------------------------------------------------------------------
TEST(FlashAttnReference, SingleKeyEqualsValue) {
  auto q = random_vec(4, 21);
  auto k = random_vec(4, 22);
  auto v = random_vec(4, 23);
  std::vector<float> o(4, 0.0f);
  auto p = make_flash_attn_params(q.data(), k.data(), v.data(), o.data(),
                                  /*batch=*/1, /*heads=*/1, /*seq_q=*/1,
                                  /*seq_k=*/1, /*head_dim=*/4,
                                  /*causal=*/false);
  reference_attention(p);
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(o[i], v[i]);
}

TEST(FlashAttnReference, UniformScoresGiveMeanValue) {
  // Q == K == 0  =>  all scores are 0  =>  softmax is uniform.
  const int nq = 3, nk = 5, d = 4;
  std::vector<float> q(static_cast<std::size_t>(nq) * d, 0.0f);
  std::vector<float> k(static_cast<std::size_t>(nk) * d, 0.0f);
  auto v = random_vec(static_cast<std::size_t>(nk) * d, 24);
  std::vector<float> o(static_cast<std::size_t>(nq) * d, 0.0f);
  auto p = make_flash_attn_params(q.data(), k.data(), v.data(), o.data(), 1, 1,
                                  nq, nk, d, /*causal=*/false);
  reference_attention(p);

  for (int i = 0; i < nq; ++i) {
    for (int dd = 0; dd < d; ++dd) {
      double mean = 0.0;
      for (int j = 0; j < nk; ++j)
        mean += v[j * d + dd];
      mean /= nk;
      EXPECT_NEAR(o[i * d + dd], mean, 1e-6) << "row " << i << " dim " << dd;
    }
  }
}

TEST(FlashAttnReference, CausalMaskDropsFutureKeys) {
  // scale == 1 for d == 1.  All scores are 0, so row i averages V[0..i]
  // (clamped to Nk).
  const int nq = 2, nk = 3, d = 1;
  std::vector<float> q(nq, 0.0f);
  std::vector<float> k(nk, 0.0f);
  std::vector<float> v = {1.0f, 3.0f, 5.0f};
  std::vector<float> o(nq, 0.0f);
  auto p = make_flash_attn_params(q.data(), k.data(), v.data(), o.data(), 1, 1,
                                  nq, nk, d, /*causal=*/true);
  reference_attention(p);
  EXPECT_NEAR(o[0], 1.0f, 1e-6);
  EXPECT_NEAR(o[1], 2.0f, 1e-6);
}

#ifdef FLASH_ATTN_USE_TORCH
TEST(FlashAttnReference, MatchesLibtorch) {
  struct TorchCase {
    int batch, heads, seq_q, seq_k, d;
    bool causal;
  };
  static const TorchCase kCases[] = {
      {1, 1, 4, 4, 8, false},
      {1, 2, 8, 8, 16, true},
      {1, 2, 32, 64, 64, false},
      {2, 3, 16, 8, 32, true},
  };

  for (const auto& c : kCases) {
    SCOPED_TRACE(::testing::Message()
                 << c.batch << "x" << c.heads << "x" << c.seq_q << "x"
                 << c.seq_k << "x" << c.d << " causal=" << c.causal);
    const std::size_t q_elems =
        static_cast<std::size_t>(c.batch) * c.heads * c.seq_q * c.d;
    const std::size_t kv_elems =
        static_cast<std::size_t>(c.batch) * c.heads * c.seq_k * c.d;
    auto q = random_vec(q_elems, 31);
    auto k = random_vec(kv_elems, 32);
    auto v = random_vec(kv_elems, 33);

    std::vector<float> o_ref(q_elems, 0.0f);
    auto p = make_flash_attn_params(q.data(), k.data(), v.data(), o_ref.data(),
                                    c.batch, c.heads, c.seq_q, c.seq_k, c.d,
                                    c.causal);
    reference_attention(p);

    auto o_torch = torch_attention(q, k, v, c.batch, c.heads, c.seq_q, c.seq_k,
                                   c.d, p.softmax_scale, c.causal);
    for (std::size_t i = 0; i < q_elems; ++i) {
      EXPECT_NEAR(o_ref[i], o_torch[i], 1e-5) << "at flat index " << i;
    }
  }
}
#endif

TEST(FlashAttnFramework, VersionRegistryIsValid) {
  const auto versions = flash_attn_versions();
  if (versions.empty()) {
    GTEST_SKIP() << kNoVersionsHint;
  }
  for (std::size_t i = 0; i < versions.size(); ++i) {
    EXPECT_NE(versions[i].name, nullptr);
    EXPECT_NE(versions[i].fn, nullptr);
    for (std::size_t j = 0; j < i; ++j) {
      if (versions[i].name != nullptr && versions[j].name != nullptr) {
        EXPECT_STRNE(versions[i].name, versions[j].name);
      }
    }
  }
}

TEST(FlashAttnFramework, RejectsInvalidParams) {
  std::vector<float> buf(64, 0.0f);
  auto make = [&] {
    return make_flash_attn_params(buf.data(), buf.data(), buf.data(),
                                  buf.data(), 1, 1, 4, 4, 4, false);
  };

  EXPECT_NO_THROW(flash_attn_check_params(make()));

  {
    auto p = make();
    p.q = nullptr;
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.batch = 0;
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.num_heads = 0;
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.seq_len_q = 0;
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.seq_len_k = 0;
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.head_dim = 0;
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.softmax_scale = 0.0f;
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.q_stride_s = 2;  // < head_dim
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.k_stride_h = 8;  // < seq_len_k * head_dim
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
  {
    auto p = make();
    p.o_stride_b = 8;  // < one head block
    EXPECT_THROW(flash_attn_check_params(p), std::invalid_argument);
  }
}

}  // namespace
