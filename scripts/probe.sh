#!/usr/bin/env bash
# 作业起不来 / 没有日志时用：到 GPU 节点上跑探针，把节点上的真实情况打印回来。
#   默认只做【阶段0】：检查节点上 /research 等路径是否可见（快，且不依赖项目目录）
#   FULL=1 scripts/probe.sh  额外做阶段1(srun 跑 probe_node.sh) 和阶段2(sbatch 提交，看日志能否生成)
#   换分区/节点:  SLURM_PARTITION=gpu_2h scripts/probe.sh    SLURM_NODELIST=projgpu39 scripts/probe.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/config.sh"
require_master || exit 1
[ "$REMOTE_MODE" = slurm ] || { echo "probe.sh 只用于 slurm 目标（TARGET=cse）"; exit 2; }

echo "==> 阶段0: 节点上的路径可见性   (slurm 参数: ${SLURM_BASE})"
{ slurm_pre; printf 'SX=%q\nRD=%q\n' "$SLURM_BASE" "$REMOTE_DIR"; cat <<'EOF'
cd "$HOME" 2>/dev/null
srun $SX -t 00:03:00 bash -c '
echo "主机: $(hostname)   用户: $(whoami)   时间: $(date +%H:%M:%S)"
for p in /research /research/d7 /research/d7/gds /research/d7/gds/sguan26 '"$RD"' "$HOME"; do
  if [ -d "$p" ]; then echo "  [ OK ] 可见: $p"; else echo "  [缺失] 看不到: $p"; fi
done
echo "--- 挂载 ---"; mount 2>/dev/null | grep -E "research|uac" | sed -n 1,4p
echo "--- /research/d7 ---"; ls /research/d7 2>&1 | sed -n 1,3p
' 2>&1
echo "srun 退出码: $?"
EOF
} | rsh

if [ "${FULL:-0}" = 1 ]; then
  rsync -az "$ROOT/scripts/" "${REMOTE_HOST}:${REMOTE_DIR}/scripts/"
  echo; echo "==> 阶段1: srun 运行 probe_node.sh"
  { slurm_pre; printf 'SX=%q\nRD=%q\n' "$SLURM_BASE" "$REMOTE_DIR"; cat <<'EOF'
cd "$RD" && mkdir -p logs results
srun $SX -t 00:03:00 bash scripts/probe_node.sh 2>&1
echo "srun 退出码: $?"
EOF
  } | rsh
  echo; echo "==> 阶段2: sbatch 提交探针（和 run.sh 同样的方式）"
  OUT=$({ slurm_pre; printf 'SX=%q\nRD=%q\n' "$SLURM_BASE" "$REMOTE_DIR"; cat <<'EOF'
cd "$RD"
sbatch $SX -t 00:03:00 -J probe -o logs/probe_%j.out --wrap 'bash scripts/probe_node.sh' 2>&1
EOF
  } | rsh)
  echo "$OUT"
  JID=$(echo "$OUT" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1)
  [ -n "$JID" ] || { echo "提交失败"; exit 1; }
  ST=UNKNOWN
  for i in $(seq 1 30); do
    ST=$({ slurm_pre; cat <<EOF
s=\$(squeue -h -j ${JID} -o '%T' 2>/dev/null | sed -n 1p)
if [ -z "\$s" ]; then s=\$(scontrol show job ${JID} 2>/dev/null | grep -o 'JobState=[A-Z_]*' | sed -n 1p | cut -d= -f2); fi
echo "\${s:-UNKNOWN}"
EOF
    } | rsh) || ST=UNKNOWN
    case "$ST" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|UNKNOWN) break;; esac
    sleep 5
  done
  echo "作业 ${JID} 状态: ${ST}"
  { slurm_pre; printf 'RD=%q\n' "$REMOTE_DIR"; cat <<EOF
cd "\$RD"
if [ -f logs/probe_${JID}.out ]; then echo "--- logs/probe_${JID}.out ---"; cat logs/probe_${JID}.out
else echo "!! 作业结束了，但 logs/probe_${JID}.out 没有生成"; fi
EOF
  } | rsh
fi
