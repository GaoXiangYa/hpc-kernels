#include "test_utils.h"
#include "add.h"
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// CPU reference — identical stride-based logic as the kernel
// ===========================================================================
static void cpu_add(const float* A, const long* strides_A,
                    const float* B, const long* strides_B,
                    float* C,       const long* strides_C,
                    const long* shape, int ndim, long total) {
    for (long idx = 0; idx < total; ++idx) {
        long off_A = 0, off_B = 0, off_C = 0;
        long rem = idx;
        for (int d = 0; d < ndim; ++d) {
            long coord = rem % shape[d];
            rem /= shape[d];
            off_A += coord * strides_A[d];
            off_B += coord * strides_B[d];
            off_C += coord * strides_C[d];
        }
        C[off_C] = A[off_A] + B[off_B];
    }
}

// ===========================================================================
// Test fixture
// ===========================================================================
class AddTest : public ::testing::Test {
protected:
    template <typename... Args>
    void run_test(std::vector<float>        A,
                  const std::vector<long>&  strides_A,
                  std::vector<float>        B,
                  const std::vector<long>&  strides_B,
                  const std::vector<long>&  shape,
                  float                     eps = 1e-4f) {
        // strides_C = contiguous (C is always a fresh output tensor)
        std::vector<long> strides_C(shape.size());
        strides_C[0] = 1;
        for (size_t d = 1; d < shape.size(); ++d)
            strides_C[d] = strides_C[d - 1] * shape[d - 1];

        long total = 1;
        for (long s : shape) total *= s;

        std::vector<float> C_ocl(total, -99.0f);
        std::vector<float> C_cpu(total, -99.0f);

        // ---------- OCL ----------
        launch_add_v1(A, strides_A, B, strides_B, C_ocl, strides_C, shape);

        // ---------- CPU ----------
        cpu_add(A.data(), strides_A.data(),
                B.data(), strides_B.data(),
                C_cpu.data(), strides_C.data(),
                shape.data(), (int)shape.size(), total);

        expect_near(C_ocl, C_cpu, eps);
    }
};

// ===========================================================================
// 1.  1D contiguous
// ===========================================================================
TEST_F(AddTest, Contiguous1D) {
    std::vector<float> A = {1, 2, 3, 4, 5};
    std::vector<float> B = {5, 4, 3, 2, 1};
    // shape = {N}, strides = {1}
    run_test(A, {1}, B, {1}, {5});
}

// ===========================================================================
// 2.  2D contiguous
// ===========================================================================
TEST_F(AddTest, Contiguous2D) {
    auto A = random_vec(6, -2, 2);   // [2][3]
    auto B = random_vec(6, -2, 2);
    // shape = {3, 2}  (innermost-first: col=3, row=2)
    // strides = {1, 3}
    run_test(A, {1, 3}, B, {1, 3}, {3, 2});
}

// ===========================================================================
// 3.  Broadcast: A[3][1] + B[1][4]  →  C[3][4]
//     A: [a₀₀, a₁₀, a₂₀]  3 elems, broadcast col dim
//     B: [b₀₀, b₀₁, b₀₂, b₀₃]  4 elems, broadcast row dim
// ===========================================================================
TEST_F(AddTest, Broadcast2D) {
    // Physical data (row-major flat)
    std::vector<float> A = {10, 20, 30};          // shape [3][1]
    std::vector<float> B = {1, 2, 3, 4};           // shape [1][4]

    // C shape = [4, 3]  (innermost = col=4, next = row=3)
    // strides_A = {0, 1}  ← col broadcast, row contiguous
    // strides_B = {1, 0}  ← col contiguous, row broadcast
    run_test(A, {0, 1}, B, {1, 0}, {4, 3});
}

// ===========================================================================
// 4.  Scalar broadcast: A[1] + B[3][2]  →  C[3][2]
// ===========================================================================
TEST_F(AddTest, ScalarBroadcast) {
    std::vector<float> A = {100};
    auto B = random_vec(6, -1, 1);
    // shape = {2, 3}, strides_A = {0, 0}, strides_B = {1, 2}
    run_test(A, {0, 0}, B, {1, 2}, {2, 3});
}

// ===========================================================================
// 5.  Non‑contiguous slice: sub‑region of a larger parent tensor
//     Parent (4×5) → slice rows=1..3, cols=1..4  →  shape [3][3]
//     Strides stay as parent's [1, 5]; offset the start pointer.
// ===========================================================================
TEST_F(AddTest, NonContiguousSlice) {
    const int PR = 4, PC = 5;
    std::vector<float> parent_A(PR * PC);
    std::vector<float> parent_B(PR * PC);

    for (int r = 0; r < PR; ++r)
        for (int c = 0; c < PC; ++c) {
            parent_A[r * PC + c] = (float)(r * 10 + c);
            parent_B[r * PC + c] = (float)(r + c);
        }

    // Slice start: row=1, col=1 → offset = 1*5 + 1 = 6
    // Use the parent vectors directly, offsetting pointers via build-time
    // We build smaller vectors by copying ONLY the needed elements,
    // then tell the kernel the correct stride so it skips gaps.
    // But stride-based access works on flat buffers.  As long as the
    // stride * (shape[d]-1) stays within the buffer, we're safe.
    // Here: max offset = (3-1)*1 + (3-1)*5 = 2 + 10 = 12
    // The buffer must be ≥ 13 elements.

    // So we copy the whole parent rows 1..3 (15 elements: rows 1,2,3 × 5 cols)
    // and rely on the kernel strides to only touch cols 1..3.
    float* row1 = parent_A.data() + 1 * PC;   // start of row 1, col 0
    std::vector<float> A(row1, row1 + 3 * PC);  // 15 elems = strong enough for stride=5 access

    float* brow1 = parent_B.data() + 1 * PC;
    std::vector<float> B(brow1, brow1 + 3 * PC);

    // shape = {3, 3} (innermost-first: col=3, row=3)
    // strides = {1, 5}  (skip 5 elements per logical row)
    run_test(A, {1, 5}, B, {1, 5}, {3, 3});
}

// ===========================================================================
// 6.  3D broadcast:  A[2][1][5] + B[1][4][5]  →  C[2][4][5]
// ===========================================================================
TEST_F(AddTest, Broadcast3D) {
    auto A = random_vec(10, -1, 1);      // physical [2][1][5]
    auto B = random_vec(20, -1, 1);      // physical [1][4][5]
    // C shape = [5, 4, 2]  (innermost = 5, then 4, then 2)
    // A: dim0 stride=1, dim1(broadcast)=0, dim2 stride=5
    // B: dim0 stride=1, dim1 stride=5,   dim2(broadcast)=0
    run_test(A, {1, 0, 5},
             B, {1, 5, 0},
             {5, 4, 2});
}
