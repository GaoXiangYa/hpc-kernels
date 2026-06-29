#include "gemm.h"
#include "device.h"

void gemm_v0(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v0.cl",
                                "gemm_v0_kernel");

  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;

  cl::NDRange global_work_size(kGlobalSizeN, kGlobalSizeM);
  cl::NDRange local_work_size(16, 16);

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v0_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v1.cl",
                                "gemm_v1_kernel");

  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;

  cl::NDRange global_work_size(kGlobalSizeM, kGlobalSizeN);
  cl::NDRange local_work_size(16, 16);

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v1_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v2(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;
  constexpr int kBlockSize = 16;

  cl::NDRange global_work_size((kGlobalSizeN + kBlockSize - 1) / kBlockSize *
                                   kBlockSize,
                               (kGlobalSizeM + kBlockSize - 1) / kBlockSize *
                                   kBlockSize);
  cl::NDRange local_work_size(kBlockSize, kBlockSize);

  const std::string build_options =
      " -funsafe-max-local-work-size=2 -expected-thread-mode=SIMD64 "
      "-external-register-sector-mode=2";

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v2.cl",
                                "gemm_v2_kernel", build_options);

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v2_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v3(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;
  constexpr int kBlockSize = 16;
  constexpr int kCoarseFactor = 4;

  int num_groups_x =
      (N + kBlockSize * kCoarseFactor - 1) / (kBlockSize * kCoarseFactor);
  int num_groups_y = (M + kBlockSize - 1) / kBlockSize;

  cl::NDRange global_work_size(num_groups_x * kBlockSize,
                               num_groups_y * kBlockSize);
  cl::NDRange local_work_size(kBlockSize, kBlockSize);

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v3.cl",
                                "gemm_v3_kernel");

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v3_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v4(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  constexpr int kVecWidth = 4;
  constexpr int kTileM = 32;
  constexpr int kTileN = 32;

  cl::NDRange local_work_size(kTileN / kVecWidth, kTileM);
  cl::NDRange global_work_size(
      (N + kTileN - 1) / kTileN * (kTileN / kVecWidth),
      (M + kTileM - 1) / kTileM * kTileM);

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v4.cl",
                                "gemm_v4_kernel");

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v4_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v5(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  constexpr int kVecWidth = 4;
  constexpr int kTileM = 32;
  constexpr int kTileN = 32;
  constexpr int kCoarseFactor = 4;

  cl::NDRange local_work_size(kTileN / kVecWidth, kTileM);
  cl::NDRange global_work_size((N + kTileN - 1) / kTileN *
                                   (kTileN / kVecWidth) / kCoarseFactor,
                               (M + kTileM - 1) / kTileM * kTileM);

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v5.cl",
                                "gemm_v5_kernel");

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v5_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v6(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  constexpr int kVecWidth = 4;
  constexpr int kTileM = 32;
  constexpr int kTileN = 32;
  constexpr int kCoarseFactor = 4;

  cl::NDRange local_work_size(kTileN / kVecWidth, kTileM);
  cl::NDRange global_work_size((N + kTileN - 1) / kTileN *
                                   (kTileN / kVecWidth) / kCoarseFactor,
                               (M + kTileM - 1) / kTileM * kTileM);

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v6.cl",
                                "gemm_v6_kernel");

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v6_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v7(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  constexpr int kRegM = 4;
  constexpr int kRegN = 4;
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;

  cl::NDRange local_work_size(kTileN, kTileM);
  cl::NDRange global_work_size(
      (N + kTileN * kRegN - 1) / (kTileN * kRegN) * kTileN,
      (M + kTileM * kRegM - 1) / (kTileM * kRegM) * kTileM);

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v7.cl",
                                "gemm_v7_kernel");

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v7_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v8(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  constexpr int kRegM = 4;
  constexpr int kRegN = 4;
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;

  cl::NDRange local_work_size(kTileN, kTileM);
  cl::NDRange global_work_size(
      (N + kTileN * kRegN - 1) / (kTileN * kRegN) * kTileN,
      (M + kTileM * kRegM - 1) / (kTileM * kRegM) * kTileM);

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v8.cl",
                                "gemm_v8_kernel");

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v8_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v9(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  constexpr int kRegM = 4;
  constexpr int kRegN = 4;
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;

  cl::NDRange local_work_size(kTileN, kTileM);
  cl::NDRange global_work_size(
      (N + kTileN * kRegN - 1) / (kTileN * kRegN) * kTileN,
      (M + kTileM * kRegM - 1) / (kTileM * kRegM) * kTileM);

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v9.cl",
                                "gemm_v9_kernel");

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v9_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}

void gemm_v10(const float* A, const float* B, float* C, int M, int N, int K,
              float alpha, float beta) {
  constexpr int kRegM = 4;
  constexpr int kRegN = 4;
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;

  cl::NDRange local_work_size(kTileN, kTileM);
  cl::NDRange global_work_size(
      (N + kTileN * kRegN - 1) / (kTileN * kRegN) * kTileN,
      (M + kTileM * kRegM - 1) / (kTileM * kRegM) * kTileM);

  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/gemm/gemm_v10.cl",
                                "gemm_v10_kernel");

  auto ba = dm.create_ro_buffer(sizeof(float) * M * K, A);
  auto bb = dm.create_ro_buffer(sizeof(float) * K * N, B);
  auto bc = dm.create_rw_buffer(sizeof(float) * M * N, C);

  kernel.setArg(0, ba);
  kernel.setArg(1, bb);
  kernel.setArg(2, bc);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size, "gemm_v10_kernel");
  dm.read_buffer(bc, sizeof(float) * M * N, C);
}
