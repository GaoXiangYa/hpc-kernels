#pragma once

// ============================================================================
// HGEMM thread organization:
//
//   block -> warp -> thread -> register
//
// GemmConfig describes the tile shape of every level.
// BlockCoord / WarpCoord / ThreadCoord describe the current hardware thread's
// position in this hierarchy.
// RegisterTile<T, Config> stores the per-thread C fragment.
// ============================================================================

template <int BlockM, int BlockN, int BlockK, int WarpM, int WarpN, int ThreadM,
          int ThreadN>
struct GemmConfig {
  // --------------------------------------------------------------------------
  // 1. Block-level tile: C[BlockM x BlockN], K is reduced in BlockK chunks.
  // --------------------------------------------------------------------------
  static constexpr int kBlockM = BlockM;
  static constexpr int kBlockN = BlockN;
  static constexpr int kBlockK = BlockK;

  // --------------------------------------------------------------------------
  // 2. Warp-level tile: each warp computes C[WarpM x WarpN].
  // --------------------------------------------------------------------------
  static constexpr int kWarpM = WarpM;
  static constexpr int kWarpN = WarpN;

  // --------------------------------------------------------------------------
  // 3. Thread-level tile: each thread computes C[ThreadM x ThreadN].
  // --------------------------------------------------------------------------
  static constexpr int kThreadM = ThreadM;
  static constexpr int kThreadN = ThreadN;

  // --------------------------------------------------------------------------
  // 4. Register-level tile: accumulator fragment owned by one thread.
  // --------------------------------------------------------------------------
  static constexpr int kRegM = kThreadM;
  static constexpr int kRegN = kThreadN;

  // --------------------------------------------------------------------------
  // Derived counts.
  // --------------------------------------------------------------------------
  static constexpr int kWarpSize = 32;

  static constexpr int kWarpsM = kBlockM / kWarpM;
  static constexpr int kWarpsN = kBlockN / kWarpN;
  static constexpr int kWarps = kWarpsM * kWarpsN;

  static constexpr int kThreadsM = kWarpM / kThreadM;
  static constexpr int kThreadsN = kWarpN / kThreadN;
  static constexpr int kThreadsPerWarp = kThreadsM * kThreadsN;
  static constexpr int kThreadsPerBlock = kWarps * kWarpSize;

  // --------------------------------------------------------------------------
  // Shared-memory leading dimensions.
  // kSmemPad is a bank-conflict tuning knob, not part of the logical tile.
  // --------------------------------------------------------------------------
  static constexpr int kSmemPad = 4;
  static constexpr int kSmemStrideA = kBlockM + kSmemPad;
  static constexpr int kSmemStrideB = kBlockN + kSmemPad;

  // --------------------------------------------------------------------------
  // Shape validation.
  // --------------------------------------------------------------------------
  static_assert(kBlockM > 0 && kBlockN > 0 && kBlockK > 0,
                "Block tile dimensions must be positive");
  static_assert(kWarpM > 0 && kWarpN > 0,
                "Warp tile dimensions must be positive");
  static_assert(kThreadM > 0 && kThreadN > 0,
                "Thread tile dimensions must be positive");

  static_assert(kBlockM % kWarpM == 0 && kBlockN % kWarpN == 0,
                "Block tile must be divisible by warp tile");
  static_assert(kWarpM % kThreadM == 0 && kWarpN % kThreadN == 0,
                "Warp tile must be divisible by thread tile");

  static_assert(kThreadsPerWarp == kWarpSize,
                "Warp tile must map exactly to 32 threads");
  static_assert(kThreadsPerBlock <= 1024, "Block must not exceed 1024 threads");
};

// ============================================================================
// Block-level coordinate.
//
// Grid convention:
//   blockIdx.x -> N dimension
//   blockIdx.y -> M dimension
// ============================================================================
template <class Config>
struct BlockCoord {
  int block_m;
  int block_n;

  // Block-local C tile origin.
  int tile_m;
  int tile_n;

  __device__ __forceinline__ BlockCoord() {
    block_m = static_cast<int>(blockIdx.y);
    block_n = static_cast<int>(blockIdx.x);

    tile_m = block_m * Config::kBlockM;
    tile_n = block_n * Config::kBlockN;
  }
};

// ============================================================================
// Warp-level coordinate inside a block.
// ============================================================================
template <class Config>
struct WarpCoord {
  int warp_id;
  int lane_id;

  int warp_m;
  int warp_n;

  // Block-local warp tile origin.
  int tile_m;
  int tile_n;

  // Global C tile origin.
  int global_m;
  int global_n;

  __device__ __forceinline__ WarpCoord(const BlockCoord<Config>& block) {
    const int tid = static_cast<int>(threadIdx.x);

    warp_id = tid / Config::kWarpSize;
    lane_id = tid % Config::kWarpSize;

    warp_m = warp_id / Config::kWarpsN;
    warp_n = warp_id % Config::kWarpsN;

    tile_m = warp_m * Config::kWarpM;
    tile_n = warp_n * Config::kWarpN;

    global_m = block.tile_m + tile_m;
    global_n = block.tile_n + tile_n;
  }
};

// ============================================================================
// Thread-level coordinate inside a warp.
// ============================================================================
template <class Config>
struct ThreadCoord {
  int thread_m;
  int thread_n;

  // Block-local thread tile origin.
  int tile_m;
  int tile_n;

  // Global C tile origin.
  int global_m;
  int global_n;

  __device__ __forceinline__ ThreadCoord(const WarpCoord<Config>& warp) {
    thread_m = warp.lane_id / Config::kThreadsN;
    thread_n = warp.lane_id % Config::kThreadsN;

    tile_m = warp.tile_m + thread_m * Config::kThreadM;
    tile_n = warp.tile_n + thread_n * Config::kThreadN;

    global_m = warp.global_m + thread_m * Config::kThreadM;
    global_n = warp.global_n + thread_n * Config::kThreadN;
  }
};

// ============================================================================
// Per-thread register tile.
//
// For HGEMM, typical instantiation is:
//   RegisterTile<float, HgemmConfig> acc;
// ============================================================================
template <typename T, class Config>
struct RegisterTile {
  static constexpr int kRows = Config::kRegM;
  static constexpr int kCols = Config::kRegN;

  T data[kRows][kCols];

  __device__ __forceinline__ T& operator()(int m, int n) { return data[m][n]; }

  __device__ __forceinline__ const T& operator()(int m, int n) const {
    return data[m][n];
  }

  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
#pragma unroll
      for (int j = 0; j < kCols; ++j) {
        data[i][j] = T(0);
      }
    }
  }
};