#!/usr/bin/env bash
# 本地一键：同步代码 -> 远程 GPU 服务器编译运行 -> 输出实时打印回本地终端。
#
# 用法:
#   scripts/remote.sh kernels/01_naive.cu [run.sh 的参数...]   同步 + 在远程运行 scripts/run.sh（最常用）
#       例: scripts/remote.sh kernels/01_naive.cu -m nsys
#           scripts/remote.sh kernels/01_naive.cu -a "data 32"
#   scripts/remote.sh sync [-n]       只同步（-n = 只预览，不真正传）
#   scripts/remote.sh sh <命令...>    同步后在远程项目目录执行任意命令
#                                     例: scripts/remote.sh sh python ref/gen_data.py 4096 1024
#   scripts/remote.sh fetch           把远程 results/ logs/ 里的 .ncu-rep .nsys-rep .log .csv 拉回本地
#   scripts/remote.sh status|up|down  查看 / 建立 / 关闭那一条被复用的 SSH 连接
#   scripts/remote.sh info            打印当前选中的机器档案与解析出的 主机 / 目录 / conda 环境（不连接）
#
# 环境变量（都有默认值）:
#   MACHINE=omen         选机器档案 machines/<名字>.env（默认 omen；旧服务器是 MACHINE=rtxpro6000）
#   REMOTE_HOST / REMOTE_DIR / CONDA_ENV   默认取自机器档案，设置同名环境变量可覆盖
#   SSH_PERSIST=8h       主连接空闲多久后自动断开
#   SYNC_DELETE=1        同步时删除远程多出来的文件（默认不删，更安全）
#   NO_SYNC=1            跳过同步
#   GPU_INDEX / NVCC_FLAGS   本地若设置了，会原样传给远程的 run.sh
#
# 连接策略: 所有 ssh / rsync 共用一条 ControlMaster 连接（socket 在 ~/.ssh/cm-*），
# 因此对服务器来说只有一次登录；ServerAlive 心跳让它能长时间保持。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 机器档案 machines/<名字>.env 提供 REMOTE_HOST / REMOTE_DIR / CONDA_ENV 的默认值（同名环境变量优先）
MACHINE="${MACHINE:-omen}"
PROFILE="$ROOT/machines/$MACHINE.env"
[ -f "$PROFILE" ] || { echo "[remote] 找不到机器档案 ${PROFILE}（可选: $(ls "$ROOT"/machines/*.env 2>/dev/null | xargs -n1 basename | sed 's/\.env$//' | tr '\n' ' ')）" >&2; exit 2; }
eval "$( . "$PROFILE"; printf 'REMOTE_HOST=%q REMOTE_DIR=%q CONDA_ENV=%q' "$REMOTE_HOST" "$REMOTE_DIR" "$CONDA_ENV" )"
SSH_PERSIST="${SSH_PERSIST:-8h}"

mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh" 2>/dev/null || true
SSH_OPTS=(
  -o ControlMaster=auto
  -o "ControlPath=$HOME/.ssh/cm-%C"
  -o "ControlPersist=$SSH_PERSIST"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
  -o ConnectTimeout=15
  -o BatchMode=yes
)

ensure_master() {
  ssh "${SSH_OPTS[@]}" -O check "$REMOTE_HOST" >/dev/null 2>&1 && return 0
  echo "[remote] 建立到 $REMOTE_HOST 的 SSH 主连接（之后所有 ssh / rsync 都复用它，不再产生新登录）" >&2
  ssh "${SSH_OPTS[@]}" -fN "$REMOTE_HOST"
}

do_sync() {
  if [ "${NO_SYNC:-0}" = 1 ]; then echo "[remote] NO_SYNC=1，跳过同步" >&2; return 0; fi
  local flags="-az -v"
  if [ "${1:-}" = "-n" ]; then flags="$flags -n"; fi
  if [ "${SYNC_DELETE:-0}" = 1 ]; then flags="$flags --delete"; fi
  echo "[remote] 同步 $ROOT/ -> $REMOTE_HOST:$REMOTE_DIR/" >&2
  # shellcheck disable=SC2086
  rsync $flags \
    --exclude=.git/ --exclude=.venv/ --exclude=data/ --exclude=results/ --exclude=logs/ \
    --exclude=__pycache__/ --exclude='*.pyc' --exclude=.DS_Store \
    --exclude='*.nsys-rep' --exclude='*.ncu-rep' --exclude='*.sqlite' \
    --rsync-path="mkdir -p $(printf %q "$REMOTE_DIR") && rsync" \
    -e "ssh ${SSH_OPTS[*]}" \
    "$ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"
}

# 在远程项目目录里执行 "$@"（按 argv 原样传递，不经过二次拆分）
remote_exec() {
  local cmd env_prefix="" v script b64
  cmd="$(printf '%q ' "$@")"
  for v in GPU_INDEX NVCC_FLAGS NCU_EXTRA FLUSH_L2 PEAK_GBPS PEAK_TFLOPS; do
    if [ -n "${!v:-}" ]; then env_prefix="$env_prefix$v=$(printf '%q' "${!v}") "; fi
  done
  read -r -d '' script <<EOF || true
source ~/.bashrc >/dev/null 2>&1 || true
if ! type conda >/dev/null 2>&1; then
  for c in ~/miniconda3 ~/anaconda3 ~/miniforge3 ~/mambaforge /opt/conda; do
    if [ -f "\$c/etc/profile.d/conda.sh" ]; then . "\$c/etc/profile.d/conda.sh"; break; fi
  done
fi
conda activate $(printf %q "$CONDA_ENV") || { echo "[remote] conda activate $CONDA_ENV 失败" >&2; exit 97; }
cd $(printf %q "$REMOTE_DIR") || { echo "[remote] 找不到目录 $REMOTE_DIR" >&2; exit 98; }
if command -v flock >/dev/null 2>&1; then
  exec 9>"/tmp/cuda_proj_\$(id -un).lock"
  flock -n 9 || { echo "[remote] 上一次运行还没结束，排队等待..."; flock -w 1800 9 || exit 99; }
fi
echo "[remote] \$(hostname) | conda env: \${CONDA_DEFAULT_ENV:-?} | nvcc: \$(command -v nvcc || echo 未找到)"
${env_prefix}${cmd}
EOF
  # 脚本用 base64 传过去，避免多层引号问题；-tt 给远程一个 pty，Ctrl-C 能传到远程进程
  b64="$(printf '%s' "$script" | base64 | tr -d '\n')"
  local t0=$SECONDS rc=0
  ssh "${SSH_OPTS[@]}" -tt "$REMOTE_HOST" "bash -ic \"\$(echo $b64 | base64 -d)\"" || rc=$?
  echo "[remote] 结束，退出码 ${rc}，耗时 $((SECONDS - t0)) 秒" >&2
  return $rc
}

do_fetch() {
  local d
  mkdir -p "$ROOT/results" "$ROOT/logs"
  for d in results logs; do
    rsync -az -v -e "ssh ${SSH_OPTS[*]}" \
      --include='*/' --include='*.ncu-rep' --include='*.nsys-rep' --include='*.log' --include='*.csv' --exclude='*' \
      "$REMOTE_HOST:$REMOTE_DIR/$d/" "$ROOT/$d/"
  done
}

case "${1:-}" in
  ""|-h|--help|help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  info)   echo "MACHINE=$MACHINE  REMOTE_HOST=$REMOTE_HOST  REMOTE_DIR=$REMOTE_DIR  CONDA_ENV=$CONDA_ENV" ;;
  up)     ensure_master; echo "[remote] 主连接已就绪" >&2 ;;
  status) if ssh "${SSH_OPTS[@]}" -O check "$REMOTE_HOST" 2>&1; then :; else echo "[remote] 当前没有活动的主连接" >&2; fi ;;
  down)   ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" 2>&1 || echo "[remote] 当前没有活动的主连接" >&2 ;;
  sync)   shift; ensure_master; do_sync "$@" ;;
  fetch)  ensure_master; do_fetch ;;
  sh)     shift; [ $# -ge 1 ] || { echo "用法: scripts/remote.sh sh <命令...>" >&2; exit 2; }
          ensure_master; do_sync; remote_exec "$@" ;;
  *)      ensure_master; do_sync; remote_exec scripts/run.sh "$@" ;;
esac
