#include "flash_attn.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Shapes are chosen to be interesting on a desktop CPU: short/long sequences,
// d = 64/128, batched, cross-attention (Nq != Nk) and single-token decode.
// ---------------------------------------------------------------------------
struct Shape {
  const char* tag;
  int batch;
  int heads;
  int seq_q;
  int seq_k;
  int head_dim;
};

const Shape kShapes[] = {
    {"tiny", 1, 1, 16, 16, 32},     {"short", 1, 8, 128, 128, 64},
    {"medium", 1, 8, 256, 256, 64}, {"long", 1, 8, 512, 512, 64},
    {"d128", 1, 8, 512, 512, 128},  {"batch", 4, 12, 128, 128, 64},
    {"cross", 1, 8, 256, 512, 64},  {"decode", 1, 8, 1, 512, 64},
};

constexpr double kTargetSeconds = 0.10;  // pick repeats to reach ~100 ms
constexpr int kMaxRepeats = 20;

// Fraction of a full Nq x Nk score matrix kept by the causal mask.
double causal_fraction(int seq_q, int seq_k) {
  const int64_t nq = seq_q;
  const int64_t nk = seq_k;
  int64_t allowed = 0;
  if (nq <= nk) {
    allowed = nq * (nq + 1) / 2;
  } else {
    allowed = nk * (nk + 1) / 2 + (nq - nk) * nk;
  }
  return static_cast<double>(allowed) /
         (static_cast<double>(nq) * static_cast<double>(nk));
}

double flops_of(const FlashAttentionParams& p) {
  // 2 * Nq*Nk*d for Q @ K^T and 2 * Nq*Nk*d for P @ V.
  double flops = 4.0 * p.batch * p.num_heads * p.seq_len_q *
                 static_cast<double>(p.seq_len_k) * p.head_dim;
  if (p.causal)
    flops *= causal_fraction(p.seq_len_q, p.seq_len_k);
  return flops;
}

// Ideal HBM/DRAM traffic: read Q, K, V once and write O once.  A real kernel
// reads K/V once per query block, so this is a lower bound (the roofline).
double ideal_bytes_of(const FlashAttentionParams& p) {
  const double nq =
      static_cast<double>(p.batch) * p.num_heads * p.seq_len_q * p.head_dim;
  const double nk =
      static_cast<double>(p.batch) * p.num_heads * p.seq_len_k * p.head_dim;
  return sizeof(float) * (2.0 * nq + 2.0 * nk);
}

template <typename Fn>
double time_ms(Fn&& fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<float> random_vec(std::size_t n, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(n);
  for (auto& x : values)
    x = dist(gen);
  return values;
}

bool version_enabled(const char* name,
                     const std::vector<std::string>& version_filter) {
  if (version_filter.empty())
    return true;  // no filter -> run everything
  return std::find(version_filter.begin(), version_filter.end(), name) !=
         version_filter.end();
}

bool shape_selected(const char* tag, const std::string& shape_filter) {
  if (shape_filter.empty())
    return true;
  std::size_t pos = 0;
  while (pos < shape_filter.size()) {
    const std::size_t comma = shape_filter.find(',', pos);
    const std::string token =
        shape_filter.substr(pos, comma == std::string::npos ? std::string::npos
                                                            : comma - pos);
    if (token == tag)
      return true;
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return false;
}

void print_usage() {
  std::printf(
      "usage: flash_attn_benchmark [vN ...] [--shape=tag1,tag2] "
      "[--list-shapes]\n");
  std::printf("shapes:");
  for (const auto& shape : kShapes)
    std::printf(" %s", shape.tag);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  // Positional arguments select versions; --shape=... selects shapes.
  std::vector<std::string> version_filter;
  std::string shape_filter;
  bool list_shapes = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--list-shapes") {
      list_shapes = true;
    } else if (arg.rfind("--shape=", 0) == 0) {
      shape_filter = arg.substr(8);
    } else {
      version_filter.push_back(arg);
    }
  }

  const auto versions = flash_attn_versions();

  std::printf("CPU flash_attn benchmark\n");
  std::printf("OpenMP max threads: %d, hardware threads: %u\n",
              omp_get_max_threads(), std::thread::hardware_concurrency());

  if (list_shapes) {
    print_usage();
    return 0;
  }

  if (versions.empty()) {
    std::printf(
        "\nno versions registered: implement src/cpu/flash_attn/"
        "flash_attn_v0.cpp and register it in flash_attn_versions.cpp\n");
    return 0;
  }

  std::printf("\n%-8s %-7s %6s %12s %6s %8s %10s %9s %8s\n", "variant", "shape",
              "BxH", "Nq x Nk", "d", "causal", "time_ms", "GFLOP/s", "GB/s");
  std::printf("%s\n", std::string(84, '-').c_str());

  for (const auto& shape : kShapes) {
    if (!shape_selected(shape.tag, shape_filter))
      continue;

    const std::size_t q_elems = static_cast<std::size_t>(shape.batch) *
                                shape.heads * shape.seq_q * shape.head_dim;
    const std::size_t kv_elems = static_cast<std::size_t>(shape.batch) *
                                 shape.heads * shape.seq_k * shape.head_dim;
    auto q = random_vec(q_elems, 101);
    auto k = random_vec(kv_elems, 102);
    auto v = random_vec(kv_elems, 103);
    std::vector<float> o(q_elems, 0.0f);

    for (int causal_i = 0; causal_i < 2; ++causal_i) {
      const bool causal = causal_i == 1;
      auto params =
          make_flash_attn_params(q.data(), k.data(), v.data(), o.data(),
                                 shape.batch, shape.heads, shape.seq_q,
                                 shape.seq_k, shape.head_dim, causal);

      for (const auto& version : versions) {
        if (!version_enabled(version.name, version_filter))
          continue;

        version.fn(params);  // warmup

        const double first = time_ms([&] { version.fn(params); });
        const int repeats = static_cast<int>(
            std::lround(kTargetSeconds * 1000.0 / std::max(first, 1e-6)));
        const int reps = std::clamp(repeats, 1, kMaxRepeats);

        double best_ms = first;
        for (int r = 1; r < reps; ++r) {
          best_ms = std::min(best_ms, time_ms([&] { version.fn(params); }));
        }

        const double gflops = flops_of(params) / (best_ms * 1e-3) / 1e9;
        const double gbps = ideal_bytes_of(params) / (best_ms * 1e-3) / 1e9;
        std::printf(
            "%-8s %-7s %3dx%-2d %5d x %-5d %4d %8d %10.3f %9.2f %8.2f\n",
            version.name, shape.tag, shape.batch, shape.heads, shape.seq_q,
            shape.seq_k, shape.head_dim, causal ? 1 : 0, best_ms, gflops, gbps);
      }
    }
  }

  std::printf(
      "\nnote: time_ms is the best of 1..%d repeats; GFLOP/s counts the causal "
      "mask exactly;\n      GB/s uses ideal traffic (Q+K+V read + O written "
      "once).\n",
      kMaxRepeats);
  return 0;
}
