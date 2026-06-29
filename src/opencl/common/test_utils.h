#pragma once

#include <gtest/gtest.h>

#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Fill a vector with uniformly-distributed random values in [lo, hi].
// ---------------------------------------------------------------------------
template <typename T>
void set_random_values(std::vector<T>& input, T lo, T hi) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(lo, hi);
    for (auto& num : input) num = dis(gen);
}

// ---------------------------------------------------------------------------
// Convenience: allocate + fill in one call.
// ---------------------------------------------------------------------------
inline std::vector<float> random_vec(size_t n, float lo = -1.0f,
                                     float hi = 1.0f) {
    std::vector<float> v(n);
    set_random_values(v, lo, hi);
    return v;
}

// ---------------------------------------------------------------------------
// Element-wise near-comparison for two float vectors.
// ---------------------------------------------------------------------------
inline void expect_near(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        float eps = 1e-3f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_NEAR(actual[i], expected[i], eps) << " at index " << i;
    }
}
