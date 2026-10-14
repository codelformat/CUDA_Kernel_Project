#!/usr/bin/env bash
# 在【当前机器】上编译并运行 CUDA 程序，可选 Nsight 分析。无需 ssh / Slurm：
# 把仓库 clone 到有 GPU 的节点上，在仓库根目录直接运行。
#
# 用法: scripts/run.sh kernels/xxx.cu [-m run|nsys|ncu|all] [-a "程序参数"] [-k ncu核函数名正则] [-c ncu抓取次数] [-s ncu set]
#   run : 编译 + 运行（默认）      nsys: + Nsight Systems 时间线
#   ncu : + Nsight Compute         all : nsys + ncu
# 环境变量: GPU_INDEX=<n>  指定显卡（默认自动选最空闲的）   NVCC_FLAGS="..."  追加 nvcc 选项
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 2; }
SRC="$1"; shift
MODE=run; ARGS=""; KERNEL=""; NCU_COUNT=5; NCU_SET=full
while getopts "m:a:k:c:s:" o; do
  case $o in
    m) MODE=$OPTARG;; a) ARGS=$OPTARG;; k) KERNEL=$OPTARG;; c) NCU_COUNT=$OPTARG;; s) NCU_SET=$OPTARG;;
    *) exit 2;;
  esac
done
[ -f "$SRC" ] || { echo "找不到 ${SRC}（路径相对仓库根目录）"; exit 2; }
case "$MODE" in run|nsys|ncu|all) ;; *) echo "-m 只能是 run|nsys|ncu|all"; exit 2;; esac
command -v nvcc >/dev/null 2>&1 || { echo "PATH 里没有 nvcc"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "找不到 nvidia-smi：这台机器没有 GPU 驱动？"; exit 1; }

NAME=$(basename "$SRC" .cu)
TAG=$(date +%Y%m%d-%H%M%S)
OUT=results/${NAME}_${TAG}
mkdir -p "$OUT" logs
BIN="$OUT/$NAME"
exec > >(tee "logs/${NAME}_${TAG}.log") 2>&1

echo "== GPU =="
hostname; date
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
GPU_INDEX=${GPU_INDEX:-auto}
if [ "$GPU_INDEX" = auto ]; then
  GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
              | sort -t, -k2,2n -k3,3n | sed -n 1p | cut -d, -f1 | tr -d ' ')
fi
export CUDA_VISIBLE_DEVICES=$GPU_INDEX
GPU_NAME=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader)
CC=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '. ' || true)
case "$CC" in ''|*[!0-9]*) CC="";; esac
read -r USED UTIL < <(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
echo "-> 使用 GPU $GPU_INDEX: $GPU_NAME  (sm_${CC:-?})  已用显存 ${USED} MiB, 利用率 ${UTIL}%"
if [ "${USED:-0}" -gt 500 ] || [ "${UTIL:-0}" -gt 10 ]; then
  echo "!! 这块卡正被别人使用，计时结果会失真（可用 GPU_INDEX=<n> 换卡，或稍后再测）"
fi

echo; echo "== 编译 =="
ARCH_FLAG="-arch=sm_${CC}"; [ -n "$CC" ] || ARCH_FLAG="-arch=native"
nvcc --version | tail -2
LIBS=""; case "$SRC" in matmul/*) LIBS="-lcublas";; esac   # matmul 的测试框架用 cuBLAS 当参考
set -x
nvcc -O3 -lineinfo -std=c++17 $ARCH_FLAG ${NVCC_FLAGS:-} "$SRC" -o "$BIN" $LIBS
set +x

if [ ! -f data/meta.txt ] && [ "$NAME" != 00_smoke_vecadd ] && [[ "$SRC" != matmul/* ]]; then
  echo "!! 没有 data/meta.txt：先生成测试数据  scripts/setup_env.sh && source .venv/bin/activate && python ref/gen_data.py"
fi

echo; echo "== 运行 =="
"$BIN" $ARGS

find_tool() {   # find_tool 名字 glob...
  local n=$1; shift
  if command -v "$n" >/dev/null 2>&1; then command -v "$n"; return 0; fi
  local g p
  for g in "$@"; do p=$(ls -d $g 2>/dev/null | sort -V | tail -1 || true); [ -n "$p" ] && { echo "$p"; return 0; }; done
  return 0
}
CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")

if [ "$MODE" = nsys ] || [ "$MODE" = all ]; then
  echo; echo "== Nsight Systems =="
  NSYS=$(find_tool nsys "$CUDA_ROOT/nsight-systems-*/bin/nsys" "/opt/nvidia/nsight-systems/*/bin/nsys")
  if [ -z "$NSYS" ]; then echo "(没有找到 nsys，跳过)"; else
    "$NSYS" profile -o "$OUT/nsys" --force-overwrite true "$BIN" $ARGS
    "$NSYS" stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$OUT/nsys.nsys-rep" || true
  fi
fi

if [ "$MODE" = ncu ] || [ "$MODE" = all ]; then
  echo; echo "== Nsight Compute (set=$NCU_SET count=$NCU_COUNT kernel=${KERNEL:-<全部>}) =="
  NCU=$(find_tool ncu "$CUDA_ROOT/nsight-compute-*/ncu" "/opt/nvidia/nsight-compute/*/ncu")
  if [ -z "$NCU" ]; then echo "(没有找到 ncu，跳过)"; else
    "$NCU" --set "$NCU_SET" ${KERNEL:+-k "regex:$KERNEL"} -c "$NCU_COUNT" -o "$OUT/ncu" --force-overwrite "$BIN" $ARGS
    "$NCU" --import "$OUT/ncu.ncu-rep" --page details | sed -n 1,200p || true
  fi
fi

echo; echo "== 完成，产物: $PWD/$OUT   日志: logs/${NAME}_${TAG}.log =="
