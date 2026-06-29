#include <cuda_runtime.h>

inline void convertToFloat4(const float *input, float4 *output, size_t N) {
  for (size_t i = 0; i < N / 4; ++i) {
    output[i] = make_float4(input[4 * i], input[4 * i + 1], input[4 * i + 2],
                            input[4 * i + 3]);
  }
}

inline void convertToFloat(const float4 *in, float *out, size_t N) {
  for (size_t i = 0; i < N; i++) {
    out[4 * i + 0] = in[i].x;
    out[4 * i + 1] = in[i].y;
    out[4 * i + 2] = in[i].z;
    out[4 * i + 3] = in[i].w;
  }
}