# matmul（GEMM）优化路线

目标：手写 `C = A · B`（行主序），从最朴素的版本一路优化到张量核心，并以 cuBLAS 为参照。
测试环境同 LayerNorm：NVIDIA RTX PRO 6000 Blackwell（sm_120），CUDA 13.2。

## 路线（每一步只改一件事，先预测、再测量、再解释）
**阶段 A：FP32 CUDA core（SIMT）**，参照 cuBLAS SGEMM（FP32 峰值 120 TFLOPS，数据手册值）
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

> 注：sm_120 使用 `mma.sync`（HMMA）这一代张量核心指令，不支持 Hopper 的 `wgmma` 和数据中心 Blackwell 的 `tcgen05`/TMEM；
> CUTLASS 在 sm_120 上的新内核主要面向块缩放低精度（NVFP4/MXFP8）。所以阶段 B、C 以 `mma.sync` 为主线。

## 运行
```bash
scripts/run.sh matmul/01_naive.cu -a "4096 4096 4096"      # run.sh 会自动链接 cuBLAS
NVCC_FLAGS="-DX_IS_ROW=1" scripts/run.sh matmul/01_naive.cu  # 用 -D 切换实验开关
```
