#!/bin/bash
#SBATCH -J cuda
#SBATCH -p gpu_24h            # 仅作兜底；run.sh 会用命令行 -p 覆盖（见 scripts/targets/cse.sh）
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH -o logs/%x_%j.out
# 由 scripts/run.sh 通过环境变量传参：SRC MODE ARGS KERNEL NCU_SET NCU_COUNT [NVCC_FLAGS]

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/cse_gpu_env.sh

: "${SRC:?SRC 未设置}"
MODE=${MODE:-run}; ARGS=${ARGS:-}; KERNEL=${KERNEL:-}; NCU_SET=${NCU_SET:-full}; NCU_COUNT=${NCU_COUNT:-5}
NAME=$(basename "$SRC" .cu)
OUT=results/${NAME}_${SLURM_JOB_ID}
mkdir -p "$OUT"
BIN="$OUT/$NAME"

echo "== 环境 =="
hostname; nvidia-smi -L; nvcc --version | tail -1
echo "python3: $(command -v python3 || echo 无)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())" 2>&1 | tail -1 || true

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo; echo "== 编译 (sm_${CC}) =="
set -x
nvcc -O3 -lineinfo -std=c++17 -arch=sm_${CC} ${NVCC_FLAGS:-} "$SRC" -o "$BIN"
set +x

echo; echo "== 运行 =="
"$BIN" $ARGS

if [ "$MODE" = nsys ] || [ "$MODE" = all ]; then
  echo; echo "== Nsight Systems =="
  nsys profile -o "$OUT/nsys" --force-overwrite true "$BIN" $ARGS
  nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$OUT/nsys.nsys-rep" || true
fi

if [ "$MODE" = ncu ] || [ "$MODE" = all ]; then
  echo; echo "== Nsight Compute (set=$NCU_SET count=$NCU_COUNT kernel=${KERNEL:-<全部>}) =="
  ncu --set "$NCU_SET" ${KERNEL:+-k "regex:$KERNEL"} -c "$NCU_COUNT" -o "$OUT/ncu" --force-overwrite "$BIN" $ARGS
  ncu --import "$OUT/ncu.ncu-rep" --page details | head -200 || true
fi

echo; echo "== 完成，产物: $SLURM_SUBMIT_DIR/$OUT =="
ls -la "$OUT"
