# Agent Note: CPU FlashAttention 算子框架

Status: implemented

## Problem

仓库的 CUDA 侧已有 `flashattention_v1`（params API、v0 baseline、测试、benchmark），
但 CPU 侧只有 gemm。需要在 `src/cpu/` 下新增 FlashAttention 框架，且**从 v0 起的
kernel 实现全部由用户自己完成**。框架要解决的问题是 kernel 之外的一整条链路：
host 参数契约、版本注册、与独立 oracle 对拍、benchmark 口径、构建集成。没有这层，
每写一版 kernel 都要重复搭测试与计时，且 CPU 侧没有现成的 benchmark 形状/指标定义。

## Decision

新增 `src/cpu/flash_attn/`，fp32、MHA、布局 `[B, H, N, d]`（row-major，
最内层 `d` 连续，stride 以元素为单位，允许 padding）。`Nq != Nk`（cross-attention）
与 causal 都支持；causal 语义固定为 **`j > i` 不参与**。GQA/MQA、fp16、
dropout 暂不支持，作为后续可选扩展。

**版本 = 普通函数**：`void flash_attn_vN(const FlashAttentionParams&)`。
它只接收 params，读 q/k/v、完整覆盖 o、不得修改输入。CPU 上不需要 CUDA 那种
"device 指针 + host 拷贝"双层 API，`make_flash_attn_params()` 直接为连续
row-major 张量填 stride，版本实现按 stride 索引即可。这样每个版本没有 host
样板代码，测试/benchmark 直接调函数。

**单一版本注册表**：`flash_attn_versions.cpp` 返回
`std::vector<FlashAttnVersion>{{"v0", flash_attn_v0}, ...}`，test 和 benchmark
都只读这一处。新增版本要改三个地方：实现 `flash_attn_vN.cpp`、在 CMake 源列表
加一行、在注册表加声明和条目。CUDA 侧目前 test 和 benchmark 各自维护一张表，
CPU 侧不再重复这个模式。

**不提供 baseline kernel**：按用户要求，v0 及之后全部由用户实现。
`flash_attn_v0.cpp` 是带 TODO 注释的占位（能编译、当前未注册），
注册表初始为空：此时所有版本用例 `GTEST_SKIP` 并提示注册位置，
benchmark 打印同样的提示后退出。这样做的好处是交付时仓库状态是"绿"的，
而用户实现 v0 后无需改测试代码即可获得全部正确性/性能反馈。

**测试 oracle**：`test.cpp` 内置 double 精度的朴素 attention（O(Nq·Nk·d)，
max-subtraction softmax），与 kernel 实现完全独立；`USE_TORCH=ON` 时再用
libtorch 交叉验证 oracle 本身（`FLASH_ATTN_USE_TORCH` 宏隔离，CPU kernel
不依赖 torch）。用例覆盖单 token、`d=1/16/32/48/64/128`、奇数长度、
`Nq != Nk`、causal 且 `Nk < Nq`、512 长序列、大数数值稳定性；输出 buffer
预填 sentinel，未完整写出会失败。默认 atol `1e-4`，512 长度放宽到 `5e-4`。

**benchmark 口径**：CPU 友好的形状表（tiny/short/medium/long/d128/batch/
cross/decode），只计函数调用时间（buffer 复用），重复次数按首测自适应到约
100 ms（1..20 次取最优）；`GFLOP/s` 的 causal 折减用精确保留元素比例
（`Σ min(i+1, Nk) / (Nq·Nk)`），不是简单乘 0.5；`GB/s` 用理想流量
（Q/K/V 各读一次 + O 写一次）作为 roofline 下界。`OMP_NUM_THREADS` 控制线程，
benchmark 打印实际线程数。

**构建**：`flash_attn` 的 leaf CMake 自带 OpenMP、GTest、可选 Torch；并修改
`src/cpu/CMakeLists.txt`，BLAS 缺失时跳过 `gemm`（打印
`CPU gemm skipped: BLAS not found`）而不是让整个 `USE_CPU=ON` configure 失败，
这样不依赖 BLAS 的 CPU 算子仍可构建。

## Verification

- 顶层 `cmake -S . -B /tmp/build_cpu_probe -DUSE_CPU=ON -DUSE_OPENCL=OFF
  -DUSE_TORCH=OFF` configure 成功（本机无 BLAS，gemm 被跳过），
  `flash_attn_test` / `flash_attn_benchmark` 构建成功；
- 交付态（注册表为空）运行：18 个用例中 4 个通过（3 个 oracle 自检 + 1 个
  参数校验）、14 个 `GTEST_SKIP`，无失败；benchmark 打印注册提示后退出；
- 用仓库外的临时 naive 实现（不提交）验证全链路：18/18 通过，benchmark 在
  全部 8 个形状 × causal 两态下输出合理的 GFLOP/s 与 GB/s；
- `USE_TORCH=ON` 构建通过，`FlashAttnReference.MatchesLibtorch` 通过；
- 新源码 `clang-format` 检查通过，`-Wall -Wextra` 无警告（torch 头文件除外）。

## Alternatives considered

- **我顺便写一个 naive v0 baseline**：能让交付即可跑/可对比，但用户明确说 v0
  起的逐步优化由他完成；写了反而是占位负担。选择留 TODO 占位。
- **等 v0 实现后再搭测试/benchmark**：把"框架"推迟到与 kernel 同时做，
  与本次任务目标相反；先交付可跳过、可自动纳入新版本的 harness。
- **只用 libtorch 做 oracle**：省代码，但测试将依赖 torch 且无法独立验证
  oracle 自身；选 double 参考 + 可选 torch 交叉验证。
- **沿用 CUDA 的 test/benchmark 各自维护版本表**：每加一版要改两处，
  容易不同步；选单一注册表。
- **给版本加 host 包装层/虚函数注册**：CPU 上没必要，函数指针数组足够，
  且实现者只关心一个 params 结构。
- **现在就支持 GQA/MQA 和 fp16**：会扩大 params 契约与测试面，而 CPU 侧
  当前没有真实需求；先 MHA/fp32。
- **BLAS 缺失时直接 configure 失败（保持现状）**：会让本机无法构建任何
  CPU 算子；选在 `src/cpu/CMakeLists.txt` 里按 BLAS 可用性跳过 gemm。

## Consequences

收益：实现新版本只需写 kernel 本身；测试/benchmark 自动覆盖；oracle 独立且
可自检；benchmark 指标与形状固定，版本间可比较；无 BLAS 环境也能构建。

代价与边界：只支持 fp32/MHA/单层 attention（无 dropout、无 GQA）；`Nq != Nk`
时 causal 语义是 `j > i`（非 decoder 常见的右下对齐），已在 README 注明；
用户每加一版需改 CMake 源列表与注册表各一行；注册表为空时测试是 skip 而非
失败，可能被误认为"已通过"（skip 信息里写明了原因）；benchmark 的 GB/s 是
理想下界，不是实测 DRAM 流量；oracle 是 O(Nq·Nk·d) double，形状很大时会拖慢
测试；无 BLAS 时 gemm 被静默跳过，需要 gemm 的用户要自己装 BLAS。

## References

- 目录说明：[src/cpu/flash_attn/README.md](../../../../src/cpu/flash_attn/README.md)
