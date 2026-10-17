# 实验环境

本项目的数据来自两台机器。**每张结果表都要标明是哪台机器测的**，两台机器的数字不能直接比较。

| | 机器 A：`rtxpro6000` | 机器 B：`omen` |
|---|---|---|
| 用在 | LayerNorm（v1–v4 全部数据） | matmul（从第 2 课起） |
| 类型 | 学校 GPU 服务器（共享） | 自己的笔记本（独占） |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition | NVIDIA GeForce RTX 3080 Ti Laptop GPU |
| 架构 / 算力版本 | Blackwell，sm_120 | Ampere，sm_86 |
| SM 数 | 188 | 58 |
| L2 缓存 | 128 MB | 4 MB |
| 显存 | 96 GB GDDR7 | 16 GB GDDR6，256 bit |
| 显存带宽（标称） | 1597 GB/s（数据手册） | 约 512 GB/s（由显存时钟 8001 MHz 推算） |
| FP32 峰值 | 120 TFLOPS（数据手册） | 约 31.2 TFLOPS（按最高 SM 时钟 2100 MHz 计算；实际持续时钟更低） |
| 每 SM 限制 | 1536 线程 / 24 块 / 100 KB 共享内存 / 65536 寄存器 | 1536 线程 / 16 块 / 100 KB 共享内存 / 65536 寄存器 |
| CUDA 工具链 | CUDA 13.2 | nvcc 12.4（系统），PyTorch 2.14.1+cu132 自带运行库 |
| 驱动 | — | 595.99.02 |
| Nsight Compute | **不可用**（无计数器权限） | **可用**（已将 `NVreg_RestrictProfilingToAdminUsers` 设为 0） |
| Nsight Systems | 可用 | 2023.4.4 |
| 主机 | — | i7-12700H，30 GB 内存，Ubuntu 26.04（宿主编译器经 `nvidia-cuda-toolkit` 指向 gcc-13） |
| 机器档案 | `machines/rtxpro6000.env` | `machines/omen.env` |

## 使用约定
- `scripts/run.sh` 会按 GPU 型号自动选中对应的机器档案，导出 `MACHINE_ID` 和理论峰值（`PEAK_GBPS`、`PEAK_TFLOPS`），测试程序的输出第一行会打印 `Machine: …`。贴结果时保留这一行。
- `scripts/remote.sh` 用 `MACHINE=omen`（默认）或 `MACHINE=rtxpro6000` 选择远程主机、目录和 conda 环境。`scripts/remote.sh info` 可以查看当前解析出的值。
- `scripts/env_check.sh` 是只读的体检脚本，换新机器时先跑一遍。

## 机器 B 的测量注意事项
- 这是笔记本 GPU，持续满载时 SM 时钟会随功耗和温度变化（空闲 1395 MHz，上限 2100 MHz）。**相对峰值的百分比只供参考，matmul 的主要指标是"相对 cuBLAS 的百分比"。**
- `nvidia-smi` 的功耗读数在这台笔记本上不可信（读出过 751 W，功耗上限显示 N/A）。判断降频看实际 SM 时钟和 `nvidia-smi -q -d PERFORMANCE`。
- 独显没有接显示器、没有其他进程占用，不受桌面程序干扰。
- 用 ncu 比较耗时时加 `NCU_EXTRA="--clock-control none"`，否则 ncu 默认把时钟锁在基础频率，耗时会偏长。
- 系统 cuBLAS（用于 matmul 测试框架的参照）与 PyTorch 自带的 cuBLAS 版本不同，GEMM 速度可能有差异；matmul 以前者为准。
