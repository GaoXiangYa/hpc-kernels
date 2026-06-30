#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "softmax.h"
#include "test_utils.h"

// ---------------------------------------------------------------------------
// LibTorch reference implementation (scaled + masked softmax along last dim)
// ---------------------------------------------------------------------------
static std::vector<float> torchSoftmax(const float* input, const float* mask,
                                       float scale, bool is_causal, int B,
                                       int H, int S_q, int S_k) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto t    = torch::from_blob(const_cast<float*>(input),
                                 {B, H, S_q, S_k}, opts)
               .clone();
    t = t * scale;

    // External additive mask  [S_q, S_k]  →  broadcast to [B,H,S_q,S_k]
    if (mask != nullptr) {
        auto m = torch::from_blob(const_cast<float*>(mask), {S_q, S_k}, opts);
        t = t + m;
    }

    // Causal mask: positions where k > q  →  –∞
    if (is_causal) {
        auto causal = torch::ones({S_q, S_k}, opts);
        causal      = torch::triu(causal, /*diagonal=*/1);
        t           = t.masked_fill(causal.to(torch::kBool),
                                    -std::numeric_limits<float>::infinity());
    }

    auto result = torch::softmax(t, /*dim=*/-1).contiguous();
    auto* ptr   = result.data_ptr<float>();
    return std::vector<float>(ptr, ptr + result.numel());
}

// ---------------------------------------------------------------------------
// Kernel function pointer type (all three variants share the same signature)
// ---------------------------------------------------------------------------
using SoftmaxFunc = void (*)(float*, const float*, const float*, int, int, int,
                             int, float, int);

static constexpr std::pair<const char*, SoftmaxFunc>
    kSoftmaxVariants[] = {
        {"v0", launchSoftmaxV0},
        {"v1", launchSoftmaxV1},
        {"v2", launchSoftmaxV2},
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class SoftmaxTest : public ::testing::Test {
protected:
    void runTest(SoftmaxFunc launchFn, const std::vector<float>& h_input,
                 const std::vector<float>& h_mask, bool use_mask, float scale,
                 bool is_causal, int B, int H, int S_q, int S_k) {
        size_t total = B * H * S_q * S_k;
        ASSERT_EQ(h_input.size(), total);

        std::vector<float> h_output(total);
        launchFn(h_output.data(), h_input.data(),
                 use_mask ? h_mask.data() : nullptr, B, H, S_q, S_k, scale,
                 is_causal);

        auto ref = torchSoftmax(h_input.data(),
                                use_mask ? h_mask.data() : nullptr, scale,
                                is_causal, B, H, S_q, S_k);
        expect_near(h_output, ref, 1e-2f);
    }
};

// =========================================================================
// Test cases – each runs against ALL THREE kernel variants via SCOPED_TRACE
// =========================================================================

// 1. simplest: single batch, single head, small sequence, no mask, no causal
TEST_F(SoftmaxTest, BasicSmallNoMask) {
    int   B = 1, H = 1, S_q = 4, S_k = 4;
    float scale = 1.0f;
    std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f,  1.0f, 2.0f, 3.0f, 4.0f,
                                -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, 1.0f, 2.0f, 0.5f};
    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, {}, false, scale, false, B, H, S_q, S_k);
    }
}

// 2. with scaling factor  (simulate d_k=8)
TEST_F(SoftmaxTest, WithScale) {
    int    B = 1, H = 2, S_q = 3, S_k = 4;
    float  scale = 1.0f / std::sqrt(8.0f);
    size_t total = B * H * S_q * S_k;
    std::vector<float> input(total);
    for (size_t i = 0; i < total; ++i)
        input[i] = (rand() % 100) / 100.0f;
    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, {}, false, scale, false, B, H, S_q, S_k);
    }
}

// 3. external mask – hide certain positions
TEST_F(SoftmaxTest, WithMask) {
    int   B = 1, H = 1, S_q = 3, S_k = 4;
    float scale = 1.0f;
    std::vector<float> input = {
        1.0f, 2.0f,  3.0f,  4.0f,   // query 0
        5.0f, 6.0f,  7.0f,  8.0f,   // query 1
        9.0f, 10.0f, 11.0f, 12.0f   // query 2
    };
    // mask  [S_q, S_k] :  –∞  means "hidden" / "not allowed to attend"
    std::vector<float> mask = {
        0.0f,       0.0f,
        -INFINITY,  0.0f,   // query 0  cannot attend to key 2
        0.0f,       -INFINITY,
        0.0f,       -INFINITY,  // query 1  cannot attend to keys 1,3
        -INFINITY,  0.0f,
        0.0f,       0.0f    // query 2  cannot attend to key 0
    };
    float neg_inf = -1e10f;
    for (auto& v : mask)
        if (std::isinf(v) && v < 0) v = neg_inf;

    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, mask, true, scale, false, B, H, S_q, S_k);
    }
}

// 4. causal mask  (upper triangle → zero probability)
TEST_F(SoftmaxTest, CausalMask) {
    int   B = 1, H = 1, S_q = 4, S_k = 4;
    float scale = 1.0f;
    std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f,  1.0f, 2.0f, 3.0f, 4.0f,
                                -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, 1.0f, 2.0f, 0.5f};
    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, {}, false, scale, true, B, H, S_q, S_k);
    }
}

// 5. multi-batch multi-head
TEST_F(SoftmaxTest, MultiBatchHead) {
    int    B = 2, H = 3, S_q = 5, S_k = 7;
    float  scale = 1.0f / std::sqrt(16.0f);
    size_t total = B * H * S_q * S_k;
    std::vector<float> input(total);
    for (size_t i = 0; i < total; ++i)
        input[i] = (rand() % 100) / 50.0f - 1.0f;
    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, {}, false, scale, false, B, H, S_q, S_k);
    }
}

// 6. different query and key lengths  (no causal)
TEST_F(SoftmaxTest, DifferentSeqLen) {
    int    B = 1, H = 2, S_q = 3, S_k = 5;
    float  scale = 1.0f / std::sqrt(10.0f);
    size_t total = B * H * S_q * S_k;
    std::vector<float> input(total);
    for (size_t i = 0; i < total; ++i)
        input[i] = (rand() % 100) / 80.0f;
    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, {}, false, scale, false, B, H, S_q, S_k);
    }
}

// 7. large dimension stress test
TEST_F(SoftmaxTest, LargeDimensions) {
    int    B = 2, H = 4, S_q = 32, S_k = 64;
    float  scale = 1.0f / std::sqrt(32.0f);
    size_t total = B * H * S_q * S_k;
    std::vector<float> input(total);
    for (size_t i = 0; i < total; ++i)
        input[i] = (rand() % 100) / 25.0f;
    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, {}, false, scale, false, B, H, S_q, S_k);
    }
}

// 8. nullptr mask – ensure no crash
TEST_F(SoftmaxTest, NullMaskPtr) {
    int   B = 1, H = 1, S_q = 2, S_k = 3;
    float scale = 1.0f;
    std::vector<float> input = {1, 2, 3, 4, 5, 6};
    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, {}, false, scale, false, B, H, S_q, S_k);
    }
}

// 9. numerical-stability check with large input values
TEST_F(SoftmaxTest, HandlingLargeValues) {
    int   B = 1, H = 1, S_q = 2, S_k = 4;
    float scale = 1.0f;
    std::vector<float> input = {
        1000.0f, 1000.0f, 1000.0f, 1000.0f,  // uniform large
        1e5f,    -1e5f,   0.0f,    0.0f
    };
    for (auto& [name, func] : kSoftmaxVariants) {
        SCOPED_TRACE(name);
        runTest(func, input, {}, false, scale, false, B, H, S_q, S_k);
    }
}
