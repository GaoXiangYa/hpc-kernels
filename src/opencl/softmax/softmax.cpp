#include "softmax.h"
#include "device.h"

void launchSoftmaxV0(float* h_output, const float* h_input, const float* h_mask,
                     int batch_size, int num_heads, int seq_q, int seq_k,
                     float scale, int is_causal) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/softmax/softmax_v0.cl",
                                "softmaxKernelV0");

  size_t elems = static_cast<size_t>(seq_q) * seq_k * batch_size * num_heads;

  const int kThreadNum = 64;
  cl::NDRange global_work_size(seq_q * kThreadNum, num_heads, batch_size);
  cl::NDRange local_work_size(kThreadNum, 1, 1);

  auto d_output = dm.create_rw_buffer(sizeof(float) * elems, h_output);
  auto d_input  = dm.create_ro_buffer(sizeof(float) * elems, h_input);
  auto d_mask   = dm.create_ro_buffer(sizeof(float) * elems, h_mask);

  kernel.setArg(0, d_output);
  kernel.setArg(1, d_input);
  kernel.setArg(2, d_mask);
  kernel.setArg(3, batch_size);
  kernel.setArg(4, num_heads);
  kernel.setArg(5, seq_q);
  kernel.setArg(6, seq_k);
  kernel.setArg(7, scale);
  kernel.setArg(8, is_causal);

  dm.launch(kernel, global_work_size, local_work_size, "softmaxKernelV0");
  dm.read_buffer(d_output, sizeof(float) * elems, h_output);
}

void launchSoftmaxV1(float* h_output, const float* h_input, const float* h_mask,
                     int batch_size, int num_heads, int seq_q, int seq_k,
                     float scale, int is_causal) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/softmax/softmax_v1.cl",
                                "softmaxKernelV1");

  size_t elems = static_cast<size_t>(seq_q) * seq_k * batch_size * num_heads;

  const int kThreadNum = 64;
  cl::NDRange global_work_size(seq_q * kThreadNum, num_heads, batch_size);
  cl::NDRange local_work_size(kThreadNum, 1, 1);

  auto d_output = dm.create_rw_buffer(sizeof(float) * elems, h_output);
  auto d_input  = dm.create_ro_buffer(sizeof(float) * elems, h_input);
  auto d_mask   = dm.create_ro_buffer(sizeof(float) * elems, h_mask);

  kernel.setArg(0, d_output);
  kernel.setArg(1, d_input);
  kernel.setArg(2, d_mask);
  kernel.setArg(3, batch_size);
  kernel.setArg(4, num_heads);
  kernel.setArg(5, seq_q);
  kernel.setArg(6, seq_k);
  kernel.setArg(7, scale);
  kernel.setArg(8, is_causal);

  dm.launch(kernel, global_work_size, local_work_size, "softmaxKernelV1");
  dm.read_buffer(d_output, sizeof(float) * elems, h_output);
}

void launchSoftmaxV2(float* h_output, const float* h_input, const float* h_mask,
                     int batch_size, int num_heads, int seq_q, int seq_k,
                     float scale, int is_causal) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/softmax/softmax_v2.cl",
                                "softmaxKernelV2");

  size_t elems = static_cast<size_t>(seq_q) * seq_k * batch_size * num_heads;

  const int kThreadNum = 64;
  cl::NDRange global_work_size(seq_q * kThreadNum, num_heads, batch_size);
  cl::NDRange local_work_size(kThreadNum, 1, 1);

  auto d_output = dm.create_rw_buffer(sizeof(float) * elems, h_output);
  auto d_input  = dm.create_ro_buffer(sizeof(float) * elems, h_input);
  auto d_mask   = dm.create_ro_buffer(sizeof(float) * elems, h_mask);

  kernel.setArg(0, d_output);
  kernel.setArg(1, d_input);
  kernel.setArg(2, d_mask);
  kernel.setArg(3, batch_size);
  kernel.setArg(4, num_heads);
  kernel.setArg(5, seq_q);
  kernel.setArg(6, seq_k);
  kernel.setArg(7, scale);
  kernel.setArg(8, is_causal);

  dm.launch(kernel, global_work_size, local_work_size, "softmaxKernelV2");
  dm.read_buffer(d_output, sizeof(float) * elems, h_output);
}
