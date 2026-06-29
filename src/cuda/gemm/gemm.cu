#include "gemm.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// a[m, n] * b[n, k] = c[m, k]
__global__ void gemm_v0_kernel(const float *a, const float *b, float *c,
                                int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < k) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}

// 使用shared memory进行优化
// a[m, n] * b[n, k] = c[m, k]
template <int BLOCK>
__global__ void gemm_v1_kernel(const float *__restrict__ matrix_a,
                                           const float *__restrict__ matrix_b,
                                           float *__restrict__ matrix_c, int m,
                                           int n, int k) {
  // dynamic allocate shared memory
  extern __shared__ float shared_mem[];
  float *shmem_a = shared_mem;
  float *shmem_b = &shared_mem[BLOCK * BLOCK];

  const int col = blockIdx.x * BLOCK + threadIdx.x;
  const int row = blockIdx.y * BLOCK + threadIdx.y;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  float sum = 0.0f;
  for (int ph = 0; ph < (n + BLOCK - 1) / BLOCK; ++ph) {

    int a_col = ph * BLOCK + tx;
    int a_idx = row * n + a_col;

    int b_row = ph * BLOCK + ty;
    int b_idx = b_row * k + col;

    if (row < m && a_col < n) {
      shmem_a[ty * BLOCK + tx] = matrix_a[a_idx];
    } else {
      shmem_a[ty * BLOCK + tx] = 0.0f;
    }

    if (b_row < n && col < k) {
      shmem_b[ty * BLOCK + tx] = matrix_b[b_idx];
    } else {
      shmem_b[ty * BLOCK + tx] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < BLOCK; ++i) {
      sum += shmem_a[ty * BLOCK + i] * shmem_b[i * BLOCK + tx];
    }

    __syncthreads();
  }

  if (row < m && col < k) {
    matrix_c[row * k + col] = sum;
  }
}

template <int BLOCK_SIZE, int COARSE_FACTOR>
__global__ void gemm_v2_kernel(
    const float *__restrict__ matrix_a, const float *__restrict__ matrix_b,
    float *__restrict__ matrix_c, int m, int n, int k) {
  extern __shared__ float shared_mem[];
  float *shmem_a = shared_mem;
  float *shmem_b = &shared_mem[BLOCK_SIZE * BLOCK_SIZE];

  int col_start = blockIdx.x * BLOCK_SIZE * COARSE_FACTOR + threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int tile_row = threadIdx.y;
  int tile_col = threadIdx.x;

  float sum[COARSE_FACTOR];

#pragma unroll
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    sum[c] = 0.0f;
  }

  for (int ph = 0; ph < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++ph) {
    int a_col = ph * BLOCK_SIZE + tile_col;
    int a_idx = row * n + a_col;

    if (row < m && a_col < n) {
      shmem_a[tile_row * BLOCK_SIZE + tile_col] = matrix_a[a_idx];
    } else {
      shmem_a[tile_row * BLOCK_SIZE + tile_col] = 0.0f;
    }

#pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int col = col_start + c * BLOCK_SIZE;
      int b_row = ph * BLOCK_SIZE + tile_row;
      int b_idx = b_row * k + col;

      if (b_row < n && col < k) {
        shmem_b[c * BLOCK_SIZE * BLOCK_SIZE + tile_col * BLOCK_SIZE +
                tile_row] = matrix_b[b_idx];
      } else {
        shmem_b[c * BLOCK_SIZE * BLOCK_SIZE + tile_col * BLOCK_SIZE +
                tile_row] = 0.0f;
      }
    }

    __syncthreads();
#pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum[c] +=
            shmem_a[tile_row * BLOCK_SIZE + i] *
            shmem_b[c * BLOCK_SIZE * BLOCK_SIZE + tile_col * BLOCK_SIZE + i];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int col = col_start + c * BLOCK_SIZE;
    if (row < m && col < k) {
      matrix_c[row * k + col] = sum[c];
    }
  }
}

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (std::bit_cast<float4 *>(&(pointer))[0])

// A[M, K] * B[K, N] = C[M, N]
// BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, shared_memory block size
// thread_size_x, thread_size_y, each thread calculates matrix[thread_size_x,
// thread_size_y]
template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K,
          const int BLOCK_SIZE_N, const int THREAD_SIZE_Y,
          const int THREAD_SIZE_X, const bool ENABLE_DOUBLE_BUFFER>
__global__ void gemm_v3_kernel(
    const float *A, const float *B, float *C, const int M, const int N,
    const int K) {
  // block idx
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // thread idx
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // the threads number in Block of x, y
  const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
  const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
  const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

  // thread id in cur block
  const int tid = ty * THREAD_X_PER_BLOCK + tx;

  // shared memory
  __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M]; // transpose A
  __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

  // registers for C
  float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
  // registers for A, B
  float frag_a[2][THREAD_SIZE_Y];
  float frag_b[2][THREAD_SIZE_X];

  // registers load global memory
  // ldg_num_a: each thread load ldg_num_a global memory numbers, use float4
  const int ldg_num_a =
      BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
  const int ldg_num_b =
      BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
  float ldg_a_reg[4 * ldg_num_a];
  float ldg_b_reg[4 * ldg_num_b];

  // threads number in one row
  const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
  const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

  // row number and col number that needs to be loaded by this thread
  const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
  const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
  const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;
// load A from global memory to shared memory
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
    int ldg_index = i / A_TILE_ROW_STRIDE * 4;
    FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(
        A[OFFSET(BLOCK_SIZE_M * by + A_TILE_ROW_START + i, A_TILE_COL, K)]);
    As[0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
    As[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
    As[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
    As[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
  }

#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
    FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(
        B[OFFSET(B_TILE_ROW_START + i, B_TILE_COL + BLOCK_SIZE_N * bx, N)]);
  }

  __syncthreads();
  // load a from shared memory to register
#pragma unroll
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
    FETCH_FLOAT4(frag_a[0][thread_y]) =
        FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
  }

#pragma unroll
  // load b from shared memory to register
  for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
    FETCH_FLOAT4(frag_b[0][thread_x]) =
        FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
  }
  __syncthreads();
  // write_stage_idx = 1, read As[0], write As[1]
  // write_stage_idx = 0, write As[1], write As[0]
  int write_stage_idx = 1;
  int tile_idx = 0;
  do {
    tile_idx += BLOCK_SIZE_K;
    // load next tile from global mem to register
    if (tile_idx < K) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
            FETCH_FLOAT4(A[OFFSET(BLOCK_SIZE_M * by + A_TILE_ROW_START + i,
                                  A_TILE_COL + tile_idx, K)]);
      }
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
            FETCH_FLOAT4(B[OFFSET(tile_idx + B_TILE_ROW_START + i,
                                  B_TILE_COL + BLOCK_SIZE_N * bx, N)]);
      }
    }

    int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
      // load next tile from shared mem to register
      // load A from shared memory to register
      for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = FETCH_FLOAT4(
            As[load_stage_idx][j + 1][THREAD_SIZE_Y * ty + thread_y]);
      }
      // load B from shared memory to register
      for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = FETCH_FLOAT4(
            Bs[load_stage_idx][j + 1][THREAD_SIZE_X * tx + thread_x]);
      }
// compute C thread_size_x * thread_size_y
#pragma unroll
      for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
          accum[thread_y][thread_x] +=
              frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
        }
      }
    }

    // load register to shared memory
    if (tile_idx < K) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_index];
        As[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_index + 1];
        As[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_index + 2];
        As[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_index + 3];
      }
// load b from global memory to shared memory
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) =
            FETCH_FLOAT4(ldg_b_reg[ldg_index]);
      }
      __syncthreads();
      write_stage_idx ^= 1;
    }

// load a from shared memory to register
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
      FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(
          As[load_stage_idx ^ 1][0][THREAD_SIZE_Y * ty + thread_y]);
    }
// load B from shared memory to register
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
      FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(
          Bs[load_stage_idx ^ 1][0][THREAD_SIZE_X * tx + thread_x]);
    }

#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
      for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
        accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
      }
    }
  } while (tile_idx < K);
// store back to C
#pragma unroll
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
      FETCH_FLOAT4(
          C[OFFSET(BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                   BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x, N)]) =
          FETCH_FLOAT4(accum[thread_y][thread_x]);
    }
  }
}

void gemm_v0(const float *a, const float *b, float *c, int m, int n,
                   int k) {
  float *dev_a = nullptr;
  auto err = cudaMalloc(&dev_a, m * n * sizeof(float));
  err = cudaMemcpy(dev_a, a, m * n * sizeof(float), cudaMemcpyHostToDevice);

  float *dev_b = nullptr;
  err = cudaMalloc(&dev_b, n * k * sizeof(float));
  err = cudaMemcpy(dev_b, b, n * k * sizeof(float), cudaMemcpyHostToDevice);

  float *dev_c = nullptr;
  err = cudaMalloc(&dev_c, m * k * sizeof(float));
  err = cudaMemcpy(dev_c, c, m * k * sizeof(float), cudaMemcpyHostToDevice);

  const int THREAD_COUNT = 32;
  dim3 block(THREAD_COUNT, THREAD_COUNT);
  dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  gemm_v0_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);

  cudaMemcpy(c, dev_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}

void gemm_v1(const float *matrix_a, const float *matrix_b,
                         float *matrix_c, int m, int n, int k) {
  float *dev_a = nullptr;
  auto err = cudaMalloc(&dev_a, m * n * sizeof(float));
  err = cudaMemcpy(dev_a, matrix_a, m * n * sizeof(float),
                   cudaMemcpyHostToDevice);

  float *dev_b = nullptr;
  err = cudaMalloc(&dev_b, n * k * sizeof(float));
  err = cudaMemcpy(dev_b, matrix_b, n * k * sizeof(float),
                   cudaMemcpyHostToDevice);

  float *dev_c = nullptr;
  err = cudaMalloc(&dev_c, m * k * sizeof(float));
  err = cudaMemcpy(dev_c, matrix_c, m * k * sizeof(float),
                   cudaMemcpyHostToDevice);

  const int BLOCK_SIZE = 32;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  // calculate shared memory size
  auto shared_mem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

  if (BLOCK_SIZE == 16) {
    gemm_v1_kernel<16>
        <<<grid, block, shared_mem_size>>>(dev_a, dev_b, dev_c, m, n, k);
  } else if (BLOCK_SIZE == 32) {
    gemm_v1_kernel<32>
        <<<grid, block, shared_mem_size>>>(dev_a, dev_b, dev_c, m, n, k);
  } else {
    gemm_v1_kernel<8>
        <<<grid, block, shared_mem_size>>>(dev_a, dev_b, dev_c, m, n, k);
  }

  cudaMemcpy(matrix_c, dev_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}

void gemm_v2(const float *matrix_a,
                                          const float *matrix_b,
                                          float *matrix_c, int m, int n,
                                          int k) {
  float *dev_a = nullptr;
  auto err = cudaMalloc(&dev_a, m * n * sizeof(float));
  err = cudaMemcpy(dev_a, matrix_a, m * n * sizeof(float),
                   cudaMemcpyHostToDevice);

  float *dev_b = nullptr;
  err = cudaMalloc(&dev_b, n * k * sizeof(float));
  err = cudaMemcpy(dev_b, matrix_b, n * k * sizeof(float),
                   cudaMemcpyHostToDevice);

  float *dev_c = nullptr;
  err = cudaMalloc(&dev_c, m * k * sizeof(float));
  err = cudaMemcpy(dev_c, matrix_c, m * k * sizeof(float),
                   cudaMemcpyHostToDevice);

  const int BLOCK_SIZE = 32;
  const int COARSE_FACTOR = 4;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((k + BLOCK_SIZE * COARSE_FACTOR - 1) / (BLOCK_SIZE * COARSE_FACTOR),
            (m + block.y - 1) / block.y);

  // calculate shared memory size
  auto shared_mem_size =
      (BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE * COARSE_FACTOR) *
      sizeof(float);

  gemm_v2_kernel<BLOCK_SIZE, COARSE_FACTOR>
      <<<grid, block, shared_mem_size>>>(dev_a, dev_b, dev_c, m, n, k);

  cudaMemcpy(matrix_c, dev_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}