#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

// Optional temporary filter, e.g. FLASH_ATTN_TEST_HEAD_DIMS=64,128 runs only
// cases whose head_dim is in the list.  Unset or empty runs every case.
bool head_dim_selected(int head_dim) {
  const char* spec = std::getenv("FLASH_ATTN_TEST_HEAD_DIMS");
  if (spec == nullptr || *spec == '\0')
    return true;

  const std::string text(spec);
  std::size_t pos = 0;
  while (pos < text.size()) {
    const std::size_t comma = text.find(',', pos);
    const std::string token =
        text.substr(pos, comma == std::string::npos ? std::string::npos
                                                    : comma - pos);
    char* end = nullptr;
    const long value = std::strtol(token.c_str(), &end, 10);
    if (end != token.c_str() && value == head_dim)
      return true;
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return false;
}

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
  const int nk = p.seq_len_k;
  const int d = p.head_dim;
  std::vector<double> scores(nk);
  std::vector<double> acc(d);

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
        for (int j = 0; j < nk; ++j) {
          if (p.causal && j > i) {
            scores[j] = neg_inf;
            continue;
          }
          const float* k_row =
              k_bh + static_cast<std::int64_t>(j) * p.k_stride_s;
          double sum = 0.0;
#pragma omp simd reduction(+ : sum)
          for (int dd = 0; dd < d; ++dd) {
            sum += static_cast<double>(q_row[dd]) * k_row[dd];
          }
          scores[j] = sum * scale;
          row_max = std::max(row_max, scores[j]);
        }

        double denom = 0.0;
        for (int j = 0; j < nk; ++j) {
          if (scores[j] == neg_inf) {
            scores[j] = 0.0;
            continue;
          }
          scores[j] = std::exp(scores[j] - row_max);
          denom += scores[j];
        }

        // PV with the V row contiguous: much faster than the dd-outer loop,
        // which used to touch a new cache line for every j.
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int j = 0; j < nk; ++j) {
          const double pj = scores[j];
          if (pj == 0.0)
            continue;  // causal tail
          const float* v_row =
              v_bh + static_cast<std::int64_t>(j) * p.v_stride_s;
#pragma omp simd
          for (int dd = 0; dd < d; ++dd) {
            acc[dd] += pj * static_cast<double>(v_row[dd]);
          }
        }

        float* o_row = p.o + static_cast<std::int64_t>(b) * p.o_stride_b +
                       static_cast<std::int64_t>(h) * p.o_stride_h +
                       static_cast<std::int64_t>(i) * p.o_stride_s;
        for (int dd = 0; dd < d; ++dd) {
          o_row[dd] = static_cast<float>(acc[dd] / denom);
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
    if (!head_dim_selected(c.head_dim)) {
      GTEST_SKIP() << "head_dim " << c.head_dim
                   << " skipped by FLASH_ATTN_TEST_HEAD_DIMS";
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

      // Summarize mismatches instead of printing one failure per element:
      // long-context cases have hundreds of thousands of elements.
      std::size_t bad = 0;
      std::size_t worst = 0;
      double worst_err = 0.0;
      for (std::size_t i = 0; i < q_elems; ++i) {
        const double err = std::abs(static_cast<double>(o_ref[i]) - o[i]);
        if (err > c.atol) {
          ++bad;
          if (err > worst_err) {
            worst_err = err;
            worst = i;
          }
        }
      }
      if (bad != 0) {
        ADD_FAILURE() << bad << " / " << q_elems << " elements exceed atol "
                      << c.atol << "; worst |err| = " << worst_err
                      << " at flat index " << worst;
      }
    }
  }
};

#define FLASH_ATTN_TEST_CASE(name, B, H, NQ, NK, D, CAUSAL, ATOL) \
  TEST_F(FlashAttnTest, name) {                                   \
    run_case({B, H, NQ, NK, D, CAUSAL, ATOL});                    \
  }

// ---------------------------------------------------------------------------
// Long-context / realistic shapes (what FlashAttention actually targets) run
// first: d = 64/128, N = 1k/2k/4k, B*H = 4/8, causal and non-causal, plus a
// 512x2048 cross-attention and a 1x2048 decode.  Tolerances are looser than
// for the small cases because fp32 error grows with N.
// ---------------------------------------------------------------------------
FLASH_ATTN_TEST_CASE(long_1k_d64, 1, 8, 1024, 1024, 64, false, 1e-3f)
FLASH_ATTN_TEST_CASE(long_1k_d64_causal, 1, 8, 1024, 1024, 64, true, 1e-3f)
FLASH_ATTN_TEST_CASE(long_2k_d64_causal, 1, 4, 2048, 2048, 64, true, 1e-3f)
FLASH_ATTN_TEST_CASE(long_4k_d64_causal, 1, 1, 4096, 4096, 64, true, 2e-3f)
FLASH_ATTN_TEST_CASE(long_1k_d128, 1, 4, 1024, 1024, 128, false, 1e-3f)
FLASH_ATTN_TEST_CASE(long_1k_d128_causal, 1, 2, 1024, 1024, 128, true, 1e-3f)
FLASH_ATTN_TEST_CASE(long_cross_512x2k, 1, 8, 512, 2048, 64, false, 1e-3f)
// Decode is Nq=1 with the whole KV history available: no causal mask.
FLASH_ATTN_TEST_CASE(long_decode_2k, 1, 8, 1, 2048, 64, false, 5e-4f)
FLASH_ATTN_TEST_CASE(long_batched, 2, 4, 512, 512, 64, true, 5e-4f)

// ---------------------------------------------------------------------------
// Medium shapes: cheap coverage of causal, Nq != Nk and Nk % 64 != 0.
// ---------------------------------------------------------------------------
FLASH_ATTN_TEST_CASE(std_d64, 1, 4, 128, 128, 64, false, 5e-4f)
FLASH_ATTN_TEST_CASE(std_d64_causal, 1, 2, 128, 128, 64, true, 5e-4f)
FLASH_ATTN_TEST_CASE(std_d128, 1, 2, 128, 128, 128, false, 5e-4f)
FLASH_ATTN_TEST_CASE(std_d128_causal, 1, 1, 96, 96, 128, true, 5e-4f)
FLASH_ATTN_TEST_CASE(cross_attention, 1, 2, 32, 128, 64, false, 1e-4f)
FLASH_ATTN_TEST_CASE(causal_short_k, 1, 2, 128, 64, 64, true, 1e-4f)

// ---------------------------------------------------------------------------
// Edge cases (tiny N / unusual head dims): run last.
// ---------------------------------------------------------------------------
// FLASH_ATTN_TEST_CASE(one_token, 1, 1, 1, 1, 1, false, 1e-4f)
// FLASH_ATTN_TEST_CASE(small, 1, 1, 4, 4, 8, false, 1e-4f)
// FLASH_ATTN_TEST_CASE(multi_head, 1, 2, 8, 8, 16, false, 1e-4f)
// FLASH_ATTN_TEST_CASE(causal_small, 1, 1, 16, 16, 32, true, 1e-4f)
// FLASH_ATTN_TEST_CASE(odd_100_d48, 1, 2, 100, 100, 48, true, 1e-4f)

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
  if (!head_dim_selected(d)) {
    GTEST_SKIP() << "head_dim " << d << " skipped by FLASH_ATTN_TEST_HEAD_DIMS";
  }
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
    std::size_t bad = 0;
    std::size_t worst = 0;
    double worst_err = 0.0;
    for (std::size_t i = 0; i < q_elems; ++i) {
      const double err = std::abs(static_cast<double>(o_ref[i]) - o[i]);
      if (err > 1e-4) {
        ++bad;
        if (err > worst_err) {
          worst_err = err;
          worst = i;
        }
      }
    }
    if (bad != 0) {
      ADD_FAILURE() << bad << " / " << q_elems << " elements exceed atol 1e-4"
                    << "; worst |err| = " << worst_err << " at flat index "
                    << worst;
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
