# 使用说明

运行环境、命令和远程脚本的详细用法。结果与分析见 [RESULTS.md](../RESULTS.md)，项目概览见 [README](../README.md)。

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
