__kernel void add_kernel(
    __global const float* A,
    __global const long*  strides_A,    // A 的各维 stride
    __global const float* B,
    __global const long*  strides_B,    // B 的各维 stride
    __global       float* C,
    __global const long*  strides_C,    // C 的各维 stride
    __global const long*  shape,        // 广播后的 shape（ndim 个元素）
    const int ndim,                      // 维度数
    const long total_elements            // shape 各维乘积
){
    long gid = get_global_id(0);
    if (gid >= total_elements) return;

    long off_A = 0, off_B = 0, off_C = 0;
    long rem = gid;

#pragma unroll
    for (int i = 0; i < ndim; ++i) {
        long coord = rem % shape[i];   // 当前维度坐标
        rem /= shape[i];                // 处理下个维度

        off_A += coord * strides_A[i];
        off_B += coord * strides_B[i];
        off_C += coord * strides_C[i];
    }

    C[off_C] = A[off_A] + B[off_B];
}
