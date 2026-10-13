# 目标：seetacloud 单机（root 直连，无 Slurm，免密）。ssh 参数在 ~/.ssh/config 的 `Host seeta`
REMOTE_MODE=direct
REMOTE_HOST="${REMOTE_HOST:-seeta}"
REMOTE_DIR="${REMOTE_DIR:-/root/autodl-tmp/cuda_proj}"   # 数据盘 autodl-tmp 下，绝对路径
GPU_INDEX="${GPU_INDEX:-auto}"
