#!/usr/bin/env bash
# 用法: scripts/run.sh kernels/xxx.cu [-m run|nsys|ncu|nvprof|all] [-a "程序参数"] [-k ncu核函数名正则] [-c ncu抓取次数] [-s ncu set]
#   默认目标机器见 scripts/config.sh；临时切换:  TARGET=cse scripts/run.sh ...
#   run: 只编译+运行 | nsys: +时间线 | ncu: +Nsight Compute(Volta 及以上) | nvprof: +旧版 profiler(Pascal 用) | all: nsys + 当前 GPU 能用的那个
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/config.sh"

[ $# -ge 1 ] || { sed -n '2,4p' "$0"; exit 2; }
SRC="$1"; shift
MODE=run; ARGS=""; KERNEL=""; NCU_COUNT=5; NCU_SET=full
while getopts "m:a:k:c:s:" o; do
  case $o in
    m) MODE=$OPTARG;; a) ARGS=$OPTARG;; k) KERNEL=$OPTARG;; c) NCU_COUNT=$OPTARG;; s) NCU_SET=$OPTARG;;
    *) exit 2;;
  esac
done
[ -f "$ROOT/$SRC" ] || { echo "找不到 ${SRC}（路径相对项目根目录）"; exit 2; }
case "$MODE" in run|nsys|ncu|nvprof|all) ;; *) echo "-m 只能是 run|nsys|ncu|nvprof|all"; exit 2;; esac
require_master || exit 1

echo "==> [1/4] 同步代码、脚本、数据（增量）-> ${REMOTE_HOST}:${REMOTE_DIR}   (target=${TARGET}, mode=${REMOTE_MODE})"
rsh <<EOF
mkdir -p $(printf %q "$REMOTE_DIR")
EOF
rsync -az --exclude '.git/' --exclude 'results/' --exclude 'logs/' --exclude '.DS_Store' --exclude '__pycache__/' --exclude ".venv/" \
  "$ROOT/" "${REMOTE_HOST}:${REMOTE_DIR}/"

mkdir -p "$ROOT/results" "$ROOT/logs"
ST=UNKNOWN

if [ "$REMOTE_MODE" = direct ]; then
  ID=$(date +%Y%m%d-%H%M%S)
  echo "==> [2/4] 在 ${REMOTE_HOST} 上直接编译并运行 (tag=${ID})"
  echo "------------------------------ 实时输出"
  set +e
  rsh <<EOF 2>&1 | tee "$ROOT/logs/direct_${ID}.out"
cd $(printf %q "$REMOTE_DIR")
mkdir -p results
export SRC=$(printf %q "$SRC") MODE=$(printf %q "$MODE") ARGS=$(printf %q "$ARGS") KERNEL=$(printf %q "$KERNEL") NCU_COUNT=$(printf %q "$NCU_COUNT") NCU_SET=$(printf %q "$NCU_SET") TAG=$(printf %q "$ID") GPU_INDEX=$(printf %q "${GPU_INDEX:-auto}")
bash scripts/node_job.sh </dev/null
EOF
  RC=${PIPESTATUS[0]}
  set -e
  echo "------------------------------"
  if [ "$RC" -eq 0 ]; then ST=COMPLETED; else ST="FAILED(exit ${RC})"; fi
  LOG="$ROOT/logs/direct_${ID}.out"
else
  echo "==> [2/4] 提交 Slurm 作业 (src=${SRC} mode=${MODE})   sbatch ${SLURM_BASE} -t ${SLURM_TIME}"
  SUBMIT_OUT=$(rsh <<EOF
$(slurm_pre)
cd $(printf %q "$REMOTE_DIR")
mkdir -p logs results
export SRC=$(printf %q "$SRC") MODE=$(printf %q "$MODE") ARGS=$(printf %q "$ARGS") KERNEL=$(printf %q "$KERNEL") NCU_COUNT=$(printf %q "$NCU_COUNT") NCU_SET=$(printf %q "$NCU_SET")
sbatch ${SLURM_BASE} -t ${SLURM_TIME} scripts/job.sbatch 2>&1
EOF
  )
  echo "$SUBMIT_OUT"
  ID=$(echo "$SUBMIT_OUT" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1)
  [ -n "$ID" ] || { echo "提交失败：没有拿到 job id（先跑 scripts/doctor.sh 排查）"; exit 1; }

  echo "==> [3/4] 等待作业 ${ID}（排队时间取决于 GPU 是否空闲）"
  last=""; unknown=0
  while :; do
    S=$(rsh <<EOF
$(slurm_pre)
s=\$(squeue -h -j $ID -o '%T|%r' 2>/dev/null | head -1)
if [ -z "\$s" ]; then s=\$(scontrol show job $ID 2>/dev/null | grep -o 'JobState=[A-Z_]*' | head -1 | cut -d= -f2); fi
echo "\${s:-UNKNOWN}"
EOF
    ) || S=UNKNOWN
    ST=${S%%|*}
    if [ "$S" != "$last" ]; then echo "    $(date +%H:%M:%S)  $S"; last="$S"; fi
    case "$ST" in
      COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE) break;;
      UNKNOWN) unknown=$((unknown+1)); if [ $unknown -ge 3 ]; then break; fi;;
      *) unknown=0;;
    esac
    sleep "$POLL_SEC"
  done
  rsync -az --include="cuda_${ID}.out" --exclude='*' "${REMOTE_HOST}:${REMOTE_DIR}/logs/" "$ROOT/logs/" || true
  LOG="$ROOT/logs/cuda_${ID}.out"
  echo "------------------------------ 作业输出: logs/cuda_${ID}.out"
  cat "$LOG" 2>/dev/null || echo "(没有找到日志)"
  echo "------------------------------"
  if [ "$ST" != COMPLETED ] || [ ! -s "$LOG" ]; then
    echo "==> 作业没有正常完成，自动诊断（scripts/why.sh ${ID}）:"
    bash "$ROOT/scripts/why.sh" "$ID" || true
    echo "------------------------------"
  fi
fi

echo "==> 拉回产物"
rsync -az --include="*_${ID}/" --include="*_${ID}/**" --exclude='*' "${REMOTE_HOST}:${REMOTE_DIR}/results/" "$ROOT/results/" || true
echo "最终状态: ${ST}    日志: ${LOG#"$ROOT"/}    产物: results/*_${ID}/"
[ "$ST" = COMPLETED ]
