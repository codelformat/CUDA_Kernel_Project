#!/usr/bin/env bash
# 体检：连接是否复用、Slurm / GPU / CUDA 情况、远端工作目录、当前登录数
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/config.sh"

echo "目标: ${TARGET}   host=${REMOTE_HOST}   mode=${REMOTE_MODE}   dir=${REMOTE_DIR}   partition=${SLURM_PARTITION}${SLURM_NODELIST:+ node=${SLURM_NODELIST}}"
echo
echo "1) SSH（必须复用 connect.sh 建立的那一条连接）"
if ! require_master; then exit 1; fi
echo "  [ OK ] 复用连接存在: $(ssh -O check "$REMOTE_HOST" 2>&1 | tr -d '\r')"

echo "2) 远端环境"
if [ "$REMOTE_MODE" = slurm ]; then
rsh <<EOF
$(slurm_pre)
echo "  主机: \$(hostname)   用户: \$(whoami)   登录shell: \$SHELL"
n=\$(who 2>/dev/null | awk -v u="\$(whoami)" '\$1==u' | wc -l)
echo "  当前登录会话数: \${n} / 5（限制每人 5 个；这里只统计带终端的会话，仅供参考）"
for c in sbatch squeue scontrol sinfo; do
  if command -v \$c >/dev/null 2>&1; then echo "  [ OK ] \$c"; else echo "  [FAIL] 找不到 \$c"; fi
done
echo "  SLURM_CONF=\${SLURM_CONF:-<未设置>}"
echo "  --- sinfo (${SLURM_PARTITION}) ---"
sinfo -p $(printf %q "$SLURM_PARTITION") -N -o '%P  节点=%N  时限=%l  GRES=%G  状态=%t' 2>&1 | head -12
EOF
else
rsh <<'EOF'
echo "  主机: $(hostname)   用户: $(whoami)   系统: $( (. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}") )"
n=$(who 2>/dev/null | awk -v u="$(whoami)" '$1==u' | wc -l); echo "  当前登录会话数: ${n}（仅供参考）"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  --- GPU ---"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version --format=csv | sed 's/^/  /'
  echo "  compute_cap（老驱动可能不支持这个字段）:"; nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1 | sed 's/^/    /' | sed -n 1,4p
else
  echo "  [FAIL] 找不到 nvidia-smi"
fi
echo "  --- CUDA 工具链 ---"
found=0
for n in $(command -v nvcc 2>/dev/null) $(ls -d /usr/local/cuda-*/bin/nvcc /usr/local/cuda/bin/nvcc 2>/dev/null); do
  found=1; echo "  $n  ->  CUDA $($n --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | sed -n 1p)"
done
[ "$found" = 1 ] || echo "  [FAIL] 没找到 nvcc"
echo "  --- profiler ---"
for t in nsys ncu nvprof; do p=$(command -v $t 2>/dev/null); echo "  $t: ${p:-PATH 中没有}"; done
EOF
fi

echo "3) 远端工作目录 ${REMOTE_DIR}"
rsh <<EOF
mkdir -p $(printf %q "$REMOTE_DIR") && touch $(printf %q "$REMOTE_DIR")/.write_test && rm -f $(printf %q "$REMOTE_DIR")/.write_test \
  && echo "  [ OK ] 可写: \$(cd $(printf %q "$REMOTE_DIR") && pwd)" || echo "  [FAIL] 不可写"
df -h $(printf %q "$REMOTE_DIR") 2>/dev/null | tail -1 | sed 's/^/  /'
EOF
