#!/usr/bin/env bash
# 在远端（root，Ubuntu/Debian）安装 nvcc / nsys / ncu / nvprof。幂等，可重复运行。
# 由 setup_seeta.sh 通过 `ssh host bash -s` 喂入；可用 CUDA_VER=12.4 之类覆盖版本。
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
say() { echo "[remote] $*"; }
die() { echo "[remote][FAIL] $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "需要 root"
# AutoDL/seetacloud 的学术加速（有则启用，装包更快）
if [ -r /etc/network_turbo ]; then set +u; source /etc/network_turbo >/dev/null 2>&1 || true; set -u; fi

# ---------- 1. GPU / 驱动 ----------
command -v nvidia-smi >/dev/null || die "nvidia-smi 不存在：驱动没装或没挂 GPU（此脚本不装驱动）"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
say "GPU: $GPU_NAME   driver: $DRV   驱动支持的最高 CUDA: ${DRV_CUDA:-?}   compute_cap: ${CC:-?}"
[ -n "$DRV_CUDA" ] || die "读不到驱动支持的 CUDA 版本"

# 选 toolkit 版本：min(12.6, 驱动支持的版本)；12.x 同时满足 Pascal(sm_61) 编译和 nvprof
pick_min() { printf '%s\n%s\n' "$1" "$2" | sort -V | head -1; }
WANT="${CUDA_VER:-$(pick_min 12.6 "$DRV_CUDA")}"
[ "$(printf '%s\n%s\n' 11.8 "$WANT" | sort -V | head -1)" = "11.8" ] || die "驱动只支持 CUDA $DRV_CUDA（<11.8），太旧，请升级驱动或换镜像"
SFX=${WANT/./-}
say "将安装 CUDA toolkit $WANT（包后缀 -$SFX）"

# ---------- 2. 系统与磁盘 ----------
. /etc/os-release
case "${ID:-}" in
  ubuntu) OSDIR="ubuntu$(echo "$VERSION_ID" | tr -d .)";;
  debian) OSDIR="debian${VERSION_ID%%.*}";;
  *) die "不支持的系统 ${ID:-?}（只支持 Ubuntu/Debian）";;
esac
case "$(uname -m)" in x86_64) RA=x86_64;; aarch64) RA=sbsa;; *) die "不支持的架构 $(uname -m)";; esac
AVAIL=$(df -BG --output=avail /usr/local | tail -1 | tr -dc 0-9)
say "系统 $OSDIR/$RA   /usr/local 可用空间 ${AVAIL}G"
[ "${AVAIL:-0}" -ge 6 ] || die "磁盘可用空间不足 6G（现有 ${AVAIL}G），请清理后重试"

# ---------- 3. 基础包 + NVIDIA apt 源 ----------
apt-get update -qq || say "apt-get update 有警告（继续）"
apt-get install -y -qq --no-install-recommends wget ca-certificates gnupg rsync build-essential || die "基础包安装失败"

if ! ls /usr/local/cuda-$WANT/bin/nvcc >/dev/null 2>&1 || ! apt-cache policy "cuda-nvcc-$SFX" 2>/dev/null | grep -q Candidate; then
  TMP=$(mktemp -d); OK=""
  for BASE in "https://mirrors.aliyun.com/nvidia-cuda/$OSDIR/$RA" "https://developer.download.nvidia.com/compute/cuda/repos/$OSDIR/$RA"; do
    say "尝试源: $BASE"
    if wget -q -T 30 -t 2 -O "$TMP/cuda-keyring.deb" "$BASE/cuda-keyring_1.1-1_all.deb"; then OK=$BASE; break; fi
  done
  [ -n "$OK" ] || die "两个 NVIDIA 源都下载不了 keyring（机器网络受限？）"
  dpkg -i "$TMP/cuda-keyring.deb" >/dev/null || die "keyring 安装失败"
  rm -f "/etc/apt/sources.list.d/cuda-$OSDIR-$RA.list"
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $OK/ /" > /etc/apt/sources.list.d/cuda-claude.list
  apt-get update -qq || say "apt-get update 有警告（继续）"
fi

# ---------- 4. 安装 ----------
say "安装 nvcc / cudart / Nsight Compute / Nsight Systems ..."
if ! apt-get install -y --no-install-recommends \
     cuda-nvcc-$SFX cuda-cudart-dev-$SFX cuda-driver-dev-$SFX cuda-nsight-compute-$SFX cuda-nsight-systems-$SFX; then
  say "精简包失败，改装完整 cuda-toolkit-$SFX"
  apt-get install -y --no-install-recommends cuda-toolkit-$SFX || die "CUDA toolkit 安装失败"
fi
say "尝试安装 nvprof（部分版本没有，失败可忽略）"
apt-get install -y --no-install-recommends cuda-nvprof-$SFX || say "cuda-nvprof-$SFX 不存在，跳过（CUDA 13 起已移除；它也不支持 sm_75 及更新的显卡）"

# ---------- 5. 让 PATH 在任何 ssh 会话里都可用 ----------
CUDA_DIR=/usr/local/cuda-$WANT
[ -d "$CUDA_DIR" ] || die "没找到 $CUDA_DIR"
if [ ! -e /usr/local/cuda ] || [ -L /usr/local/cuda ]; then ln -sfn "$CUDA_DIR" /usr/local/cuda; fi
link_tool() {   # link_tool 名字
  local n=$1 p
  p=$(ls "$CUDA_DIR/bin/$n" 2>/dev/null | head -1)
  [ -n "$p" ] || p=$(find "$CUDA_DIR" /opt/nvidia -maxdepth 5 -name "$n" \( -type f -o -type l \) -perm -u+x 2>/dev/null | sort -V | tail -1)
  if [ -n "$p" ]; then ln -sfn "$p" "/usr/local/bin/$n"; say "链接 $n -> $p"; else say "未找到 $n"; fi
}
for t in nvcc ncu nsys nvprof; do link_tool $t; done
cat > /etc/profile.d/cuda.sh <<EOP
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
EOP

# ---------- 6. 自检 ----------
echo; say "================ 版本自检 ================"
for t in nvcc nsys ncu nvprof; do
  if command -v $t >/dev/null 2>&1; then
    printf '%-7s %s\n' "$t" "$($t --version 2>&1 | grep -iE 'release|version' | head -1)"
  else printf '%-7s %s\n' "$t" "（未安装）"; fi
done
W=$(mktemp -d); cat > "$W/t.cu" <<'EOK'
#include <cstdio>
__global__ void k(float* p){ p[threadIdx.x] = threadIdx.x * 2.f; }
int main(){ float* d; cudaMalloc(&d, 128*4); k<<<1,128>>>(d); cudaError_t e = cudaDeviceSynchronize(); printf("kernel: %s\n", cudaGetErrorString(e)); return e != cudaSuccess; }
EOK
nvcc -O2 -arch=native "$W/t.cu" -o "$W/t" && "$W/t" && say "最小 kernel 编译+运行 OK"
if command -v ncu >/dev/null 2>&1; then
  ncu --section SpeedOfLight -c 1 "$W/t" 2>&1 | grep -E "ERR_NVGPUCTRPERM|Duration|==PROF== Disconnected" | head -3 \
    && say "ncu 自检已执行（若出现 ERR_NVGPUCTRPERM，说明宿主机禁止普通进程读 GPU 计数器，容器内无法自行修复）"
fi
rm -rf "$W"
say "安装完成"
