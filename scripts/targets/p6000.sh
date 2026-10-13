# 目标：4 卡 Quadro P6000 节点（Pascal, sm_61），经跳板机 ssh 直连，没有 Slurm
# ssh 参数（HostName / Port / ProxyJump 等）都在你本机 ~/.ssh/config 的 `Host 4p6000` 里
REMOTE_MODE=direct
REMOTE_HOST="${REMOTE_HOST:-4p6000}"
REMOTE_DIR="${REMOTE_DIR:-workspace/cuda_proj}"   # 相对远端 $HOME，即 ~/workspace/cuda_proj
GPU_INDEX="${GPU_INDEX:-auto}"                    # auto = 自动选最空闲的一块；也可指定 0 / 1 / 2 / 3
