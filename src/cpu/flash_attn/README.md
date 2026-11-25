# CPU FlashAttention

CPU 版 FlashAttention 框架（fp32、MHA、`[B, H, N, d]` 布局）。各版本 kernel 由你从
v0 开始逐步实现，这里提供：host 侧参数契约、版本注册表、对拍测试（double 参考 +
可选 libtorch 交叉验证）、benchmark、以及 CPU 侧优化路线。

## 数学定义

```
S = scale * Q @ K^T                 # [Nq, Nk]
P = softmax(S)                      # causal: j > i 的元素不参与
O = P @ V                           # [Nq, d]
```

- `scale` 默认 `1 / sqrt(head_dim)`，也可通过 `make_flash_attn_params` 显式传入；
- 张量全部 fp32、row-major：`q [B,H,Nq,d]`、`k/v [B,H,Nk,d]`、`o [B,H,Nq,d]`；
- `Nq != Nk`（cross-attention）以及 `causal` 同时支持；causal 语义固定为
  **`j > i` 不参与**（i 是 query 行号，从 0 开始）；
- stride 单位是元素个数，覆盖 batch / head / row；最内层 `head_dim` 约定连续
  （row 内 stride = 1），允许带 padding 的非连续 tensor。

## 文件

| 文件 | 作用 |
| --- | --- |
| `flash_attn.h` | `FlashAttentionParams`、参数契约、host helper、版本注册表接口 |
| `flash_attn.cpp` | 连续张量 params 构造 + 参数/stride 校验 |
| `flash_attn_versions.cpp` | **版本注册表**：唯一需要登记版本的地方 |
| `flash_attn_v0.cpp` | v0 占位文件（TODO，当前未注册；由你实现） |
| `test.cpp` | double 精度朴素参考 + 可选 libtorch 交叉验证 + 变体参数化测试 |
| `benchmark.cpp` | 典型形状的 kernel-only 计时，GFLOP/s + 有效带宽 |
| `CMakeLists.txt` | 静态库 + gtest + benchmark，OpenMP 已链好 |

## 版本契约

一个版本就是：

```cpp
void flash_attn_vN(const FlashAttentionParams& params);
```

要求：

1. 用 `params.*_stride_*` 索引，不要假设张量连续；
2. 读 `q/k/v`，**完整写出 `o`**（被 mask 的行也要写，写 0 或 V 的加权和均可）；
3. 不修改 `q/k/v`；
4. `scale` 用 `params.softmax_scale`，causal 用 `params.causal`；
5. 建议在函数入口调用一次 `flash_attn_check_params(params)`（框架会抛
   `std::invalid_argument`，不触碰数据）。

## 新增一个版本（v0 开始）

以 v0 为例：

1. 在 `flash_attn_v0.cpp` 里实现函数体（注释里已写好 TODO 和提示）；
2. 把 `flash_attn_v0.cpp` 加进 `CMakeLists.txt` 的 `flash_attn_cpu` 源文件列表
   （v0 已在列表里，新增 v1/v2... 时要加一行）；
3. 在 `flash_attn_versions.cpp` 里打开声明并登记：

   ```cpp
   void flash_attn_v0(const FlashAttentionParams&);
   ...
   std::vector<FlashAttnVersion> flash_attn_versions() {
     return {
         {"v0", flash_attn_v0},
     };
   }
   ```

之后 `test.cpp` 的所有 shape 和 `benchmark.cpp` 的所有 shape 都会自动跑该版本，
不需要再改测试或 benchmark。

## 构建与运行

```bash
cmake -S . -B build -DUSE_CPU=ON -DUSE_OPENCL=OFF
cmake --build build --target flash_attn_test flash_attn_benchmark -j
cd build
./src/cpu/flash_attn/flash_attn_test
./src/cpu/flash_attn/flash_attn_benchmark          # 跑注册表里全部版本
./src/cpu/flash_attn/flash_attn_benchmark v0 v1    # 只跑指定版本
```

- benchmark 的重复次数按第一次计时自适应，目标约 100 ms/项；
- 多线程版本用 `OMP_NUM_THREADS` 控制线程数，benchmark 会打印实际线程数；
- 只编译不跑：`cmake --build build --target flash_attn_test flash_attn_benchmark`。

> 说明：`src/cpu` 后端里既有的 `gemm` 依赖 BLAS。没有 BLAS 时
> `src/cpu/CMakeLists.txt` 会跳过 gemm（configure 输出
> `CPU gemm skipped: BLAS not found`），只构建 flash_attn；需要 gemm 时装好
> BLAS 重新 configure 即可。

## 测试口径

- `test.cpp` 的参考实现是 **double 精度**的朴素 attention（O(Nq·Nk·d)，
  max-subtraction softmax），与 kernel 实现完全独立；
- 额外用 libtorch（`USE_TORCH=ON` 时）作为第二个 oracle 交叉验证参考实现本身；
- 用例覆盖：单 token、小 shape、多头、`d = 1/16/32/48/64/128`、奇数长度、
  `Nq != Nk`、causal 且 `Nk < Nq`、`N = 512` 长序列、大数输入（数值稳定性）、
  输出 sentinel 检查（未写满输出会失败）；
- 默认绝对误差 `1e-4`，512 长序列放宽到 `5e-4`；
- 注册表为空时所有版本用例 `GTEST_SKIP`，参考实现自检照常运行。

## CPU 优化路线（建议顺序）

- [ ] **v0**：最直的逐行 softmax + 加权 V，先跑通链路（当前占位）
- [ ] **v1**：KV 分块 + online softmax（单遍、不物化 `Nq×Nk`），每块重标定
- [ ] **v2**：`float`/`double` 分块矩阵乘，手工展开 + FMA（`-mavx2 -mfma` 已由
      顶层 CMake 打开）
- [ ] **v3**：AVX2 向量化：`d` 方向向量化 + 4/8 行 query 一起处理（提高 FMA 密度）
- [ ] **v4**：OpenMP 并行：按 `(b, h, q-block)` 分发；注意 `Nq` 小（decode）时
      并行度不足，可改按 KV 分块并行
- [ ] **v5**：多线程 + SIMD 组合，调 block size（`BLOCK_M/BLOCK_N`）匹配 L2
- [ ] **v6**：按 `head_dim` 做编译期模板特化、K 转置布局、减少 `exp` 调用
- [ ] 可选：GQA/MQA（K/V head 共享）、fp16/bf16、dropout

## 性能口径

- `GFLOP/s = 4·B·H·Nq·Nk·d / t`（causal 按精确保留比例折减，不是简单 ×0.5）；
- `GB/s` 按理想流量 `(Q + K + V 各读一次 + O 写一次)` 计算，是 roofline 下界；
  flash attention 实际会多次读 K/V，真实流量更大；
- 计时只包含 kernel 函数调用（输入输出 buffer 复用，不重新分配）。

## 设计记录

为什么用共享注册表、为什么参考实现放 double、为什么这个骨架不提供 baseline
kernel：[Agent Note](../../../.agents/notes/implemented/feature/2026-11-25-cpu-flash-attn-scaffold.md)
