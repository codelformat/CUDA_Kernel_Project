# 目标：CSE GPU 集群（经 linux5 提交 Slurm 作业）
REMOTE_MODE=slurm
REMOTE_HOST="${REMOTE_HOST:-cse}"
# 工作目录放在远端 home 下（相对 $HOME，即 /uac/gds/sguan26/cuda_kernel_proj）。
# 旧位置 /research/d7/gds/sguan26/... 曾在 projgpu33(gpu_2h) 上看不到；projgpu39(gpu_24h) 能看到 /research，
# 想改回 /research 时:  REMOTE_DIR=/research/d7/gds/sguan26/cuda_kernel_proj scripts/run.sh ...
REMOTE_DIR="${REMOTE_DIR:-cuda_kernel_proj}"
REMOTE_SLURM_CONF="${REMOTE_SLURM_CONF:-}"

# Slurm 参数（run.sh / probe.sh 都用这里；命令行上的 -p 会覆盖 job.sbatch 里写的 #SBATCH -p）
SLURM_PARTITION="${SLURM_PARTITION:-gpu_24h}"   # gpu_24h -> projgpu39，能看到挂载目录；旧的 gpu_2h 会落到 projgpu33 等
SLURM_NODELIST="${SLURM_NODELIST:-}"            # 想钉死节点时填，例如 projgpu39（留空 = 分区内任选）
SLURM_TIME="${SLURM_TIME:-01:00:00}"            # 单个作业时限（gpu_24h 允许更长）
# 额外的 sbatch/srun 参数，例如:  SBATCH_EXTRA="--exclude=projgpu33" scripts/run.sh ...
SBATCH_EXTRA="${SBATCH_EXTRA:-}"
