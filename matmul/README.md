# matmul（GEMM）优化路线

目标：手写 `C = A · B`（行主序），从最朴素的版本一路优化到张量核心，并以 cuBLAS 为参照。

## 实验环境
**机器 B**（`machines/omen.env`）：HP OMEN 17 笔记本，NVIDIA GeForce RTX 3080 Ti Laptop GPU（Ampere，sm_86，58 SM，4 MB L2，16 GB GDDR6，约 512 GB/s），
nvcc 12.4，Nsight Compute 可用。FP32 理论峰值约 31 TFLOPS（按最高时钟 2100 MHz），但笔记本的持续时钟更低，所以**主要指标是相对 cuBLAS 的百分比**。
详见 [docs/ENVIRONMENTS.md](../docs/ENVIRONMENTS.md)。**本目录的每张结果表都要写明机器和 CUDA 版本。**（LayerNorm 用的是另一台机器 A，数字不可比。）

## 路线（每一步只改一件事，先预测、再测量、再解释）
**阶段 A：FP32 CUDA core（SIMT）**，参照 cuBLAS SGEMM
1. `01_naive`：一个 thread 一个输出元素
2. 合并访存（线程到输出元素的映射）
3. 共享内存分块（tiling）
4. 寄存器分块（每个 thread 算一小块输出）
5. 向量化访存、共享内存无冲突、双缓冲

**阶段 B：张量核心**（fp16/bf16 输入，fp32 累加），参照 cuBLAS 的张量核心 GEMM
6. WMMA API：第一次用张量核心
7. `mma.sync` PTX + `ldmatrix`：直接控制 16×8×16 的小矩阵乘
8. 共享内存 swizzle（消除 bank 冲突）
9. `cp.async` 多级流水线
10. 分块大小 / 占用率 / split-K 调优

**阶段 C：CUTLASS / CuTe**
11. 用 CUTLASS 的 Ampere 风格（`mma.sync`）GEMM 对照手写版
12. CuTe 的 Layout / TiledMMA / TiledCopy，读懂 CUTLASS 怎么选分块与流水线

> 注：sm_86（Ampere）的张量核心指令就是 `mma.sync`，配合 `ldmatrix` 和 `cp.async`，也是 CUTLASS 的 Ampere 风格 GEMM 所针对的架构，
> 所以阶段 B、C 以它们为主线。（据 GA10x 白皮书推算，GeForce Ampere 的 FP16 张量核心在 FP32 累加下峰值约为 FP32 CUDA core 的 2 倍，待用 cuBLAS 实测确认。）

## 进度
| 课 | 内容 | 机器 | 最好结果（1024³，相对 cuBLAS） |
|---|---|---|---|
| 1–2 | naive；线程映射与合并访存 | 机器 B，CUDA 12.4 | 11.5% |
| 3 | 共享内存分块（TILE 32 / 16） | 机器 B，CUDA 12.4 | 14.9% |
| 4 | 寄存器分块（一维） | 机器 B，CUDA 12.4 | 17.2% |
| 5 | 寄存器分块（二维 8×8） | 机器 B，CUDA 12.4 | 36.8% |

详细数据与分析见 [RESULTS.md](RESULTS.md)。

## 运行
```bash
scripts/run.sh matmul/01_naive.cu -a "4096 4096 4096"      # run.sh 会自动链接 cuBLAS
NVCC_FLAGS="-DX_IS_ROW=1" scripts/run.sh matmul/01_naive.cu  # 用 -D 切换实验开关
```
