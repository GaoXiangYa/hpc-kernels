#define MIN(a, b) a < b ? a : b
#define MAX(a, b) a < b ? b : a

__kernel void softmaxKernelV2(
    __global float* restrict output,        // [B][H][S_q][S_k]
    const __global float* restrict input,   // [B][H][S_q][S_k]
    const __global float* restrict mask,    // [S_q][S_k]  (NULL when unused)
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    float scale,
    int is_causal
) {
  int group0 = get_group_id(0);   // query index  (0 .. seq_q-1)
  int group1 = get_group_id(1);   // head index   (0 .. num_heads-1)
  int group2 = get_group_id(2);   // batch index  (0 .. batch_size-1)
  int lid    = get_local_id(0);
  int local_size = get_local_size(0);

  int offset = group0 * seq_k
             + group1 * seq_q * seq_k
             + group2 * seq_k * seq_q * num_heads;

  const __global float* pin  = input  + offset;
  __global float*       pout = output + offset;

  const int vec_len    = seq_k / 4;          // number of full float4 blocks
  const int tail_start = vec_len * 4;        // first scalar tail index

  // =====================================================================
  // Stage 1 – find max
  // =====================================================================
  float lmax = -INFINITY;

  // float4 vectorized
  for (int i = lid; i < vec_len; i += local_size) {
    float4 val = vload4(i, pin) * scale;

    if (mask) {
      val.x += mask[group0 * seq_k + i * 4 + 0];
      val.y += mask[group0 * seq_k + i * 4 + 1];
      val.z += mask[group0 * seq_k + i * 4 + 2];
      val.w += mask[group0 * seq_k + i * 4 + 3];
    }
    if (is_causal) {
      int base = i * 4;
      if (base + 0 > group0) val.x = -INFINITY;
      if (base + 1 > group0) val.y = -INFINITY;
      if (base + 2 > group0) val.z = -INFINITY;
      if (base + 3 > group0) val.w = -INFINITY;
    }
    lmax = MAX(lmax, val.x);
    lmax = MAX(lmax, val.y);
    lmax = MAX(lmax, val.z);
    lmax = MAX(lmax, val.w);
  }

  // scalar tail
  for (int i = tail_start + lid; i < seq_k; i += local_size) {
    float val = pin[i] * scale;
    if (mask)    val += mask[group0 * seq_k + i];
    if (is_causal && i > group0) val = -INFINITY;
    lmax = MAX(lmax, val);
  }

  float max_v = sub_group_reduce_max(lmax);

  // =====================================================================
  // Stage 2 – exp  +  sum
  // =====================================================================
  float lsum = 0.0f;

  // float4 vectorized
  for (int i = lid; i < vec_len; i += local_size) {
    float4 val = vload4(i, pin) * scale;

    if (mask) {
      val.x += mask[group0 * seq_k + i * 4 + 0];
      val.y += mask[group0 * seq_k + i * 4 + 1];
      val.z += mask[group0 * seq_k + i * 4 + 2];
      val.w += mask[group0 * seq_k + i * 4 + 3];
    }

    float4 exp_val;
    int    base = i * 4;
    exp_val.x = (is_causal && base + 0 > group0) ? 0.0f : exp(val.x - max_v);
    exp_val.y = (is_causal && base + 1 > group0) ? 0.0f : exp(val.y - max_v);
    exp_val.z = (is_causal && base + 2 > group0) ? 0.0f : exp(val.z - max_v);
    exp_val.w = (is_causal && base + 3 > group0) ? 0.0f : exp(val.w - max_v);

    vstore4(exp_val, i, pout);
    lsum += exp_val.x + exp_val.y + exp_val.z + exp_val.w;
  }

  // scalar tail
  for (int i = tail_start + lid; i < seq_k; i += local_size) {
    float val = pin[i] * scale;
    if (mask)    val += mask[group0 * seq_k + i];
    float e = (is_causal && i > group0) ? 0.0f : exp(val - max_v);
    pout[i] = e;
    lsum += e;
  }

  float inv_sum = 1.0f / sub_group_reduce_add(lsum);

  // =====================================================================
  // Stage 3 – normalize
  // =====================================================================
  for (int i = lid; i < vec_len; i += local_size) {
    float4 v = vload4(i, pout);
    vstore4(v * inv_sum, i, pout);
  }
  for (int i = tail_start + lid; i < seq_k; i += local_size) {
    pout[i] *= inv_sum;
  }
}
