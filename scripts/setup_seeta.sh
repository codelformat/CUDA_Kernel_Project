#!/usr/bin/env bash
# 一键：接入 seeta 目标 -> 远端安装 nvcc/nsys/ncu/nvprof -> 用 01_naive.cu 验证编译与 profiling
# 用法:  scripts/setup_seeta.sh        （全程无需交互；日志在 logs/setup_seeta_*.log）
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
# 防止 Mac 空闲休眠打断长时间安装
if [ -z "${_SEETA_WRAPPED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export _SEETA_WRAPPED=1; exec caffeinate -i bash "$0" "$@"
fi
LOG="$ROOT/logs/setup_seeta_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export TARGET=seeta
source "$ROOT/scripts/config.sh"

echo "==> [1/5] ssh 配置（Host seeta + 连接复用）"
mkdir -p ~/.ssh && touch ~/.ssh/config
if grep -qE '^Host[[:space:]]+seeta([[:space:]]|$)' ~/.ssh/config; then
  echo "    ~/.ssh/config 已有 Host seeta，跳过"
else
  { echo; cat "$ROOT/scripts/ssh_config_seeta.snippet"; } >> ~/.ssh/config
  echo "    已追加 Host seeta 到 ~/.ssh/config"
fi

echo "==> [2/5] 建立复用连接（免密）"
bash "$ROOT/scripts/connect.sh" || { echo "!! 连不上 ${REMOTE_HOST}"; exit 1; }

echo "==> [3/5] 远端安装 CUDA 工具（失败自动重试，最多 3 次）"
INSTALL=FAIL
for i in 1 2 3; do
  if { echo "CUDA_VER=${CUDA_VER:-}"; cat "$ROOT/scripts/remote_install_cuda.sh"; } | rsh; then INSTALL=OK; break; fi
  echo "    第 $i 次失败，10 秒后重试"; sleep 10
done
[ "$INSTALL" = OK ] || { echo "!! 安装失败，日志: ${LOG#"$ROOT"/}"; exit 1; }

echo "==> [4/5] 体检"
bash "$ROOT/scripts/doctor.sh" || true

echo "==> [5/5] 用 01_naive.cu 验证编译 + profiling"
bash "$ROOT/scripts/run.sh" kernels/01_naive.cu -m all;    R_ALL=$?
bash "$ROOT/scripts/run.sh" kernels/01_naive.cu -m nvprof; R_NVP=$?

echo
echo "================ 总结 ================"
echo "安装:                 $INSTALL"
echo "编译+运行+nsys+ncu:   $([ $R_ALL -eq 0 ] && echo OK || echo "FAILED($R_ALL)")"
echo "nvprof:               $([ $R_NVP -eq 0 ] && echo OK || echo "FAILED($R_NVP)（nvprof 不支持 sm_75 及更新显卡，属正常）")"
echo "日志: ${LOG#"$ROOT"/}      产物: results/01_naive_*/"
echo "以后日常使用:  TARGET=seeta scripts/run.sh kernels/01_naive.cu -m all"
