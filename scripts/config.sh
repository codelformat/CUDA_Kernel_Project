# 项目配置。选择目标机器:   TARGET=p6000 scripts/run.sh ...      （默认 cse）
# 各目标的参数在 scripts/targets/<名字>.sh 里。
_CFG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${TARGET:-cse}"
if [ ! -f "${_CFG_DIR}/targets/${TARGET}.sh" ]; then
  echo "未知 TARGET=${TARGET}，可选: $(ls "${_CFG_DIR}/targets" | sed 's/\.sh$//' | tr '\n' ' ')" >&2
  exit 2
fi
source "${_CFG_DIR}/targets/${TARGET}.sh"
REMOTE_DIR="${REMOTE_DIR#\~/}"     # 容忍写成 ~/workspace/xxx
REMOTE_SLURM_CONF="${REMOTE_SLURM_CONF:-}"
POLL_SEC="${POLL_SEC:-10}"

# sbatch / srun 的公共参数（只对 slurm 目标有意义）
SLURM_PARTITION="${SLURM_PARTITION:-gpu_24h}"
SLURM_NODELIST="${SLURM_NODELIST:-}"
SLURM_TIME="${SLURM_TIME:-01:00:00}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"
SLURM_BASE="-p ${SLURM_PARTITION} --qos gpu --account gpu --gres=gpu:1 -c 4${SLURM_NODELIST:+ -w ${SLURM_NODELIST}}${SBATCH_EXTRA:+ ${SBATCH_EXTRA}}"

# 通过 stdin 把脚本喂给远端 bash 执行（避免远端登录 shell 不同带来的引号 / export 问题）
rsh() { ssh "$REMOTE_HOST" bash -s; }
slurm_pre() { if [ -n "$REMOTE_SLURM_CONF" ]; then printf 'export SLURM_CONF=%q\n' "$REMOTE_SLURM_CONF"; fi; }

# 登录节点限制每人最多 5 个登录。所有脚本都必须复用 connect.sh 建立的【那一条】SSH 连接；
# 如果它不存在就直接拒绝，绝不自己另开新连接（否则每次调用都可能多占一个登录名额）。
require_master() {
  if ! ssh -O check "$REMOTE_HOST" >/dev/null 2>&1; then
    echo "没有可复用的 SSH 连接（${REMOTE_HOST}）。请先运行 scripts/connect.sh。" >&2
    echo "（登录节点最多允许 5 个登录，脚本只在复用那一条连接时才会运行。）" >&2
    return 1
  fi
}
