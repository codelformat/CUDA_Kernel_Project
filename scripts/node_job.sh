#!/bin/bash
# 直连 GPU 节点（无 Slurm）：选卡 -> 选 CUDA -> 编译 -> 运行 -> 可选 profile
# 由 scripts/run.sh 经 ssh 触发，工作目录 = 远端项目根目录。
# 环境变量: SRC TAG [MODE ARGS KERNEL NCU_SET NCU_COUNT GPU_INDEX NVCC_FLAGS]
set -eo pipefail
: "${SRC:?SRC 未设置}"; : "${TAG:?TAG 未设置}"
MODE=${MODE:-run}; ARGS=${ARGS:-}; KERNEL=${KERNEL:-}; NCU_SET=${NCU_SET:-full}; NCU_COUNT=${NCU_COUNT:-5}
GPU_INDEX=${GPU_INDEX:-auto}
NAME=$(basename "$SRC" .cu)
OUT=results/${NAME}_${TAG}
mkdir -p "$OUT"
BIN="$OUT/$NAME"

echo "== 主机与 GPU =="
hostname; date
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
if [ "$GPU_INDEX" = auto ]; then
  GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
              | sort -t, -k2,2n -k3,3n | sed -n 1p | cut -d, -f1 | tr -d ' ')
fi
export CUDA_VISIBLE_DEVICES=$GPU_INDEX
GPU_NAME=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader)
CC=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '. ' || true)
case "$CC" in ''|*[!0-9]*) CC="";; esac        # 老驱动不支持 compute_cap 字段时会得到乱码，清空
if [ -z "$CC" ]; then
  case "$GPU_NAME" in *P6000*|*P5000*|*"GTX 10"*|*"TITAN X"*) CC=61;; esac
fi
read -r USED UTIL < <(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
echo "-> 使用 GPU $GPU_INDEX: $GPU_NAME  (compute capability: ${CC:-未知})  已用显存 ${USED} MiB, 利用率 ${UTIL}%"
if [ "${USED:-0}" -gt 500 ] || [ "${UTIL:-0}" -gt 10 ]; then
  echo "!! 这块卡正被别人使用，计时结果会失真（可用 GPU_INDEX 指定别的卡，或稍后再测）"
fi

echo; echo "== CUDA 工具链 =="
cuda_major() { "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | sed -n 1p; }
NEED_OLD=0
if [ -n "$CC" ] && [ "$CC" -lt 75 ]; then NEED_OLD=1; fi    # CUDA 13 起不再支持 sm_75 以下
cands=()
if command -v nvcc >/dev/null 2>&1; then cands+=("$(command -v nvcc)"); fi
for d in $(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -rV || true); do
  if [ -x "${d}bin/nvcc" ]; then cands+=("${d}bin/nvcc"); fi
done
if [ -x /usr/local/cuda/bin/nvcc ]; then cands+=(/usr/local/cuda/bin/nvcc); fi
NVCC=""
for c in "${cands[@]}"; do
  m=$(cuda_major "$c")
  if [ "$NEED_OLD" = 1 ] && [ "${m:-99}" -ge 13 ]; then echo "跳过 $c (CUDA ${m}.x 不支持 sm_${CC})"; continue; fi
  NVCC=$c; break
done
if [ -z "$NVCC" ]; then
  echo "找不到可用的 nvcc。sm_${CC:-?} 需要 CUDA 12.x 或更早的版本；请确认节点上装了哪些 CUDA。"
  exit 1
fi
export PATH="$(dirname "$NVCC"):$PATH"
CUDA_ROOT=$(dirname "$(dirname "$NVCC")")
echo "nvcc: $NVCC"; "$NVCC" --version | tail -2

find_tool() {   # find_tool 名字 额外glob...
  local n=$1; shift
  if command -v "$n" >/dev/null 2>&1; then command -v "$n"; return 0; fi
  local g p
  for g in "$@"; do
    p=$(ls -d $g 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$p" ]; then echo "$p"; return 0; fi
  done
  return 0
}

ARCH_FLAG="-arch=sm_${CC}"
if [ -z "$CC" ]; then ARCH_FLAG="-arch=native"; fi
echo; echo "== 编译 ($ARCH_FLAG) =="
set -x
nvcc -O3 -lineinfo -std=c++17 $ARCH_FLAG ${NVCC_FLAGS:-} "$SRC" -o "$BIN"
set +x

echo; echo "== 运行 =="
"$BIN" $ARGS

# ---- profile ----
KPROF=ncu
if [ -n "$CC" ] && [ "$CC" -lt 70 ]; then KPROF=nvprof; fi    # Nsight Compute 不支持 Pascal
case "$MODE" in
  run)    WANT="";;
  nsys)   WANT="nsys";;
  ncu)    WANT="ncu";;
  nvprof) WANT="nvprof";;
  all)    WANT="nsys $KPROF";;
  *)      echo "未知 MODE=$MODE"; exit 2;;
esac
has() { case " $WANT " in *" $1 "*) return 0;; esac; return 1; }

if has nsys; then
  echo; echo "== Nsight Systems =="
  NSYS=$(find_tool nsys "$CUDA_ROOT/nsight-systems-*/bin/nsys" "/opt/nvidia/nsight-systems/*/bin/nsys" "/opt/nvidia/nsight-systems/*/target-linux-x64/nsys")
  if [ -z "$NSYS" ]; then
    echo "(没有找到 nsys，跳过)"
  else
    "$NSYS" profile -o "$OUT/nsys" --force-overwrite true "$BIN" $ARGS
    REP=$(ls "$OUT"/nsys.nsys-rep "$OUT"/nsys.qdrep 2>/dev/null | sed -n 1p || true)
    if [ -n "$REP" ]; then
      "$NSYS" stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$REP" || "$NSYS" stats "$REP" || true
    fi
  fi
fi

if has ncu; then
  echo; echo "== Nsight Compute =="
  if [ -n "$CC" ] && [ "$CC" -lt 70 ]; then
    echo "!! GPU 是 sm_${CC}（Pascal）：Nsight Compute 在 2019.5.1 之后不再支持 Pascal，已跳过。请改用 -m nvprof。"
  else
    NCU=$(find_tool ncu "$CUDA_ROOT/nsight-compute-*/ncu" "/opt/nvidia/nsight-compute/*/ncu")
    if [ -z "$NCU" ]; then
      echo "(没有找到 ncu，跳过)"
    else
      "$NCU" --set "$NCU_SET" ${KERNEL:+-k "regex:$KERNEL"} -c "$NCU_COUNT" -o "$OUT/ncu" --force-overwrite "$BIN" $ARGS
      "$NCU" --import "$OUT/ncu.ncu-rep" --page details | sed -n 1,200p || true
    fi
  fi
fi

if has nvprof; then
  echo; echo "== nvprof（Pascal 可用的老牌 profiler；CUDA 13 起已移除）=="
  if ! command -v nvprof >/dev/null 2>&1; then
    echo "(当前 CUDA 里没有 nvprof，跳过)"
  else
    nvprof "$BIN" $ARGS 2>&1 || true
    echo "--- 指标: 占用率与显存吞吐 ---"
    nvprof --metrics achieved_occupancy,dram_read_throughput,dram_write_throughput "$BIN" $ARGS 2>&1 || true
  fi
fi

echo; echo "== 完成，产物: $PWD/$OUT =="
ls -la "$OUT"
