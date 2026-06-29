#include <cuda_runtime.h>
#include <functional>
#include <iostream>

template <typename KernelFunc, typename... Args>
void benchmarkKernel(const char *name, dim3 grid, dim3 block, double flops,
                     double bytes, int repeat, KernelFunc kernel,
                     Args... args) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warm-up
  kernel<<<grid, block>>>(args...);
  cudaDeviceSynchronize();

  // timing
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    kernel<<<grid, block>>>(args...);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= repeat;

  double seconds = ms / 1000.0;
  double gflops = (flops / seconds) / 1e9;
  double bandwidth = (bytes / seconds) / 1e9;

  std::cout << "==== Benchmark: " << name << " ====\n";
  std::cout << "Kernel time: " << ms << " ms\n";
  if (flops > 0)
    std::cout << "Throughput: " << gflops << " GFLOPS\n";
  if (bytes > 0)
    std::cout << "Memory BW:  " << bandwidth << " GB/s\n";
  std::cout << "=============================\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
