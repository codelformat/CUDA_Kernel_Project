# CUDA Kernel Project — LayerNorm 从 naive 到优化

手写 CUDA LayerNorm kernel（fp32，N=4096，H=1024），从最朴素的版本逐步优化，每一步只改一件事，并以 PyTorch 的正确性和性能作为对照。

## 结果一览（RTX PRO 6000 Blackwell，冷 L2，256 线程，中位耗时）
| 版本 | 本轮改动 | 耗时 | 相对 PyTorch |
|---|---|---|---|
| v1 `01_naive` | 一个 thread 一行，三遍扫描 | 0.560 ms（32 线程最佳 0.267 ms） | 5.5% |
| v2 `02_block_per_row` | 一个 block 一行：合并访存 + 共享内存树形归约 | 0.040 ms | 78% |
| v3 `03_regcache` | x 只读一次，缓存在寄存器 | 0.032 ms | 97% |
| v4 `04_vec4` | `float4` 向量化 load/store（128 位访存） | **0.029 ms** | **107%** |

对照：PyTorch `F.layer_norm` 0.031 ms；纯拷贝 kernel（带宽上限）0.028 ms，v4 达到其 97%。v1 → v4 同口径（256 线程）约 19 倍。所有版本与 PyTorch 的 `max_abs_err < 1e-5`。
每一轮遇到的问题、怎么发现、怎么改，以及测量方法（冷 L2、计时分辨率）、验证证据和局限见 [RESULTS.md](RESULTS.md)。

## 目录
| 路径 | 内容 |
|---|---|
| `kernels/00_smoke_vecadd.cu` | 环境冒烟测试（SAXPY） |
| `kernels/01_naive.cu` | v1：每行一个线程，三遍扫描（均值 → 方差 → 输出） |
| `kernels/02_block_per_row.cu` | v2：每行一个 block，合并访存 + 共享内存树形归约 |
| `kernels/03_regcache.cu` | v3：x 只读一次，缓存在寄存器（模板常量保证不 spill） |
| `kernels/04_vec4.cu` | v4：`float4` 向量化 load/store |
| `kernels/common.cuh` | 各版本共用的测试框架：正确性校验、热身、计时、冷 L2（`FLUSH_L2=1`）、有效带宽 |
| `kernels/block_reduce.cuh` | block 内求和归约（v2~v4 共用） |
| `kernels/bench_copy.cu` | 纯拷贝 kernel，作为带宽上限的参照 |
| `ref/bench_torch.py` | PyTorch `F.layer_norm` 的计时参照（eager / CUDA graph，热 / 冷 L2） |
| `RESULTS.md` | 各版本的数据、方法、验证证据、结论与局限 |
| `ref/layernorm_ref.py` | NumPy 参考实现（有偏方差，ε=1e-5） |
| `ref/gen_data.py` | 用 PyTorch 生成输入与参考输出到 `data/`（不入库） |
| `scripts/run.sh` | 本机编译 + 运行 + 可选 Nsight 分析 |
| `scripts/setup_env.sh` | 创建 `.venv`（numpy + torch），用于生成数据 |

## 在 GPU 机器上使用
需要：`nvcc`、`nvidia-smi`（可选 `nsys` / `ncu`）。

```bash
git clone https://github.com/codelformat/CUDA_Kernel_Project.git && cd CUDA_Kernel_Project

# 1. 生成测试数据（N=4096 行, H=1024 列），写入 data/
scripts/setup_env.sh && source .venv/bin/activate
python ref/gen_data.py

# 2. 编译并运行（自动按 GPU 选 -arch=sm_XX），打印与 PyTorch 的 max_abs_err
scripts/run.sh kernels/01_naive.cu

# 2b. 冷 L2 计时（先把 L2 冲掉，才是真正的显存带宽），指定 256 线程/block
FLUSH_L2=1 scripts/run.sh kernels/04_vec4.cu -a "data 256"
NVCC_FLAGS="-Xptxas -v" scripts/run.sh kernels/04_vec4.cu         # 看每个 thread 的寄存器数与 spill

# 3. 分析
scripts/run.sh kernels/01_naive.cu -m nsys                        # Nsight Systems
scripts/run.sh kernels/01_naive.cu -m ncu -k layernorm -c 1       # Nsight Compute，只抓 layernorm 开头的 kernel
scripts/run.sh kernels/01_naive.cu -m all                         # 两者都跑
GPU_INDEX=1 scripts/run.sh kernels/01_naive.cu                    # 多卡机器上指定显卡
```
产物在 `results/<kernel>_<时间>/`，日志在 `logs/`（均不入库）。之后更新代码用 `git pull`。

## 从本地一键远程运行（推荐）
本地改代码，用 `scripts/remote.sh` 同步到 GPU 服务器、远程编译运行，输出实时打印回本地终端。
需要：`~/.ssh/config` 里有免密登录的主机别名（默认 `4p6000`）。所有 ssh / rsync 复用**同一条** SSH 连接（ControlMaster），对服务器只有一次登录。

```bash
scripts/remote.sh kernels/01_naive.cu                 # 同步 + 远程 scripts/run.sh
scripts/remote.sh kernels/01_naive.cu -m nsys         # 参数原样传给 run.sh
scripts/remote.sh kernels/01_naive.cu -a "data 32"
scripts/remote.sh sync -n                             # 只预览会同步哪些文件
scripts/remote.sh sh python ref/gen_data.py           # 在远程项目目录跑任意命令
scripts/remote.sh fetch                               # 拉回 .ncu-rep / .nsys-rep / .log
scripts/remote.sh status | down                       # 查看 / 关闭复用的连接
```
默认值可用环境变量覆盖：`REMOTE_HOST` `REMOTE_DIR` `CONDA_ENV`（见脚本头部注释）。本地是唯一的代码来源，不要在服务器上直接改被同步的文件。

## 正确性标准
float32 下与 `torch.nn.functional.layer_norm` 的最大绝对误差 < 1e-5。

## 备注
早期版本曾通过 ssh + Slurm 从本机远程提交作业（CUHK CSE GPU 集群）；现在改为直接在 GPU 节点上运行，相关脚本已移除，仍可在 git 历史（提交 `aa7259c`）中找到。
