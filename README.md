# CUDA LayerNorm 优化之路

本地写代码，一条命令在 CSE GPU（RTX 3090, sm_86）上编译、运行、profile，结果自动拉回本地。

## 一次性准备
```bash
cat scripts/ssh_config.snippet >> ~/.ssh/config   # 加入 cse 别名与连接复用
scripts/connect.sh                                # 输入一次密码，之后 8 小时免密
scripts/doctor.sh                                 # 体检
```

## 日常使用
```bash
scripts/run.sh kernels/00_smoke_vecadd.cu              # 编译 + 运行
scripts/run.sh kernels/xxx.cu -m nsys                  # + Nsight Systems
scripts/run.sh kernels/xxx.cu -m ncu -k layernorm -c 2 # + Nsight Compute，只抓名字含 layernorm 的前 2 次
scripts/run.sh kernels/xxx.cu -a "4096 1024"           # 传程序参数
```
报告在 `results/<名字>_<jobid>/`，用本机 Nsight Compute (>=2024.3.2) / Nsight Systems (>=2024.5.1) 打开。

## 目录
- `kernels/` 每一版 kernel 一个文件（v0 朴素 -> v1 -> ...，保留历史方便对比）
- `ref/` NumPy / PyTorch 参考实现
- `scripts/` 自动化脚本    - `results/` `logs/` 拉回的产物（不进 git）
- 大文件放远端 `/research/d7/gds/sguan26`（1TB 配额）

## 切换目标机器
目标机器的参数在 `scripts/targets/<名字>.sh`，默认 `cse`，临时切换用环境变量：
```bash
scripts/run.sh kernels/01_naive.cu -m all              # 默认目标 cse（Slurm，RTX 3090，Nsight Compute 可用）
TARGET=p6000 scripts/run.sh kernels/01_naive.cu        # 4 卡 P6000 节点（直连 ssh，无 Slurm）
TARGET=p6000 GPU_INDEX=2 scripts/run.sh kernels/01_naive.cu   # 指定 P6000 节点上的第 3 块卡
```
- p6000 目标使用你 ~/.ssh/config 里已有的 `Host 4p6000`（含跳板机）。首次使用：
  然后 `scripts/connect.sh` -> `scripts/doctor.sh` -> `scripts/push_data.sh`。
- Pascal（P6000, sm_61）限制：Nsight Compute 不支持，用 `-m nvprof`；CUDA 13 不能编译 sm_61，脚本会自动选 12.x。
- 多人共享的 GPU 节点计时会受别人影响；脚本会自动选最空闲的卡并在被占用时给出警告。

## CSE 登录节点的资源限制（登录时的提示）
- 每人最多 **5 个登录**：脚本只复用 `connect.sh` 建立的那一条 SSH 连接（没有就拒绝运行），所以脚本本身只占 1 个名额。
  自己同时开的终端请保持在 3 个以内；多余的及时 `exit`。
- 每人最多 1024 个进程（Linux 里线程也算）：一次只提交一个作业，不要并行起很多任务。
- 每个会话常驻内存 6GB：登录节点上只做提交和同步，编译、运行、profile 都在 GPU 节点上（`run.sh` 就是这么做的）。

## 每次运行都会重新上传
`run.sh` 每次都会用 rsync 把 `kernels/`、`scripts/`、`ref/` 和 `data/` 增量同步到远端工作目录，再编译运行；
所以远端的代码永远和本地一致。数据没变时 rsync 几乎不花时间。CSE 目标的工作目录是远端 home 下的 `cuda_kernel_proj`。

## 分区 / 节点（CSE 集群）
默认提交到 `gpu_24h`（projgpu39，能看到 /research 挂载）。在 `scripts/targets/cse.sh` 改默认值，或临时覆盖：
```
SLURM_PARTITION=gpu_2h scripts/run.sh kernels/01_naive.cu     # 回到 2 小时分区
SLURM_NODELIST=projgpu39 scripts/run.sh kernels/01_naive.cu   # 钉死节点
SLURM_TIME=02:00:00 scripts/run.sh ...                        # 作业时限（默认 01:00:00）
scripts/probe.sh                                              # 先确认节点上路径可见
```

## seeta 目标（seetacloud 单机，root 直连）
```bash
scripts/setup_seeta.sh                                   # 一次性：写 ssh 配置、远端装 nvcc/nsys/ncu/nvprof、用 01_naive.cu 验证
TARGET=seeta scripts/run.sh kernels/01_naive.cu -m all   # 之后日常使用
# 代码同步到远端 /root/autodl-tmp/cuda_proj（run.sh 每次自动增量 rsync）
```
