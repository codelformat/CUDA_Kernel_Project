#!/usr/bin/env bash
# 新机器体检（只读，不改任何系统设置）：GPU / 驱动 / CUDA / Nsight / conda / CUTLASS / 网络 / 时钟与功耗。
# 用法（在能 ssh 到目标机器的电脑上，仓库根目录）:
#   ssh omen 'bash -s' < scripts/env_check.sh 2>&1 | tee env_report.txt
# 或在目标机器上直接:  bash scripts/env_check.sh
set +e
sec() { echo; echo "===== $* ====="; }
have() { command -v "$1" >/dev/null 2>&1; }

sec "系统"
hostname; whoami; uname -srm
(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")
grep -m1 "model name" /proc/cpuinfo; echo "逻辑核数: $(nproc)"; free -h | sed -n 2p
echo "WSL? $(grep -qi microsoft /proc/version && echo 是 || echo 否)"
echo "sudo 免密? $(sudo -n true 2>/dev/null && echo 是 || echo 否)"

sec "GPU（nvidia-smi）"
if ! have nvidia-smi; then echo "没有 nvidia-smi"; else
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total,persistence_mode,display_active,display_mode --format=csv
  echo "-- 功耗 / 温度 / 时钟（当前 与 上限）"
  nvidia-smi --query-gpu=power.draw,power.limit,power.max_limit,temperature.gpu,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,pstate --format=csv
  echo "-- 正在使用 GPU 的进程"
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
  echo "-- 降频原因（Idle 以外为 Active 说明在限速）"
  nvidia-smi -q -d PERFORMANCE | sed -n '/Clocks Event Reasons\|Clocks Throttle Reasons/,/^$/p' | head -20
  echo "-- 时钟是否被锁定"
  nvidia-smi -q -d CLOCK | grep -iE "locked|applications|default app" | head
fi

sec "性能计数器权限（决定 ncu 能不能用）"
grep -iE "RmProfilingAdminOnly" /proc/driver/nvidia/params 2>/dev/null || echo "读不到 /proc/driver/nvidia/params"
echo "(RmProfilingAdminOnly: 1 = 只有管理员/root 能读计数器；0 = 普通用户也能)"

sec "CUDA 工具链"
for t in nvcc ncu nsys cuobjdump nvdisasm compute-sanitizer cmake g++ git gh python3; do
  printf "%-18s %s\n" "$t" "$(command -v $t || echo 未找到)"
done
have nvcc && nvcc --version | tail -2
have ncu && ncu --version | tail -1
have nsys && nsys --version
have g++ && g++ --version | head -1
ls -d /usr/local/cuda* 2>/dev/null
echo "-- cuBLAS 库:"; (ldconfig -p 2>/dev/null | grep -m3 libcublas; ls /usr/local/cuda*/lib64/libcublas.so* 2>/dev/null | head -3)
have gh && { echo "-- gh 登录状态:"; gh auth status 2>&1 | head -5; }

sec "编译并运行一个小程序：设备参数 + 能否抓 ncu 计数器"
if have nvcc; then
  T=$(mktemp -d)
  cat > $T/t.cu <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
static int attr(cudaDeviceAttr a) { int v = 0; cudaDeviceGetAttribute(&v, a, 0); return v; }
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("name=%s  cc=%d.%d  SMs=%d  显存=%.1f GB  位宽=%d bit\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, attr(cudaDevAttrGlobalMemoryBusWidth));
    printf("L2=%d MB  每SM: 线程=%d 块=%d 共享内存=%d KB 寄存器=%d   每block共享内存上限(opt-in)=%d KB\n",
           attr(cudaDevAttrL2CacheSize) >> 20, attr(cudaDevAttrMaxThreadsPerMultiProcessor),
           attr(cudaDevAttrMaxBlocksPerMultiprocessor), attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor) >> 10,
           attr(cudaDevAttrMaxRegistersPerMultiprocessor), attr(cudaDevAttrMaxSharedMemoryPerBlockOptin) >> 10);
    int n = 1 << 24; float *x, *y; cudaMalloc(&x, n * 4); cudaMalloc(&y, n * 4);
    cudaMemset(x, 0, n * 4); cudaMemset(y, 0, n * 4);
    saxpy<<<(n + 255) / 256, 256>>>(n, 2.f, x, y);
    cudaError_t e = cudaDeviceSynchronize();
    printf("kernel: %s\n", cudaGetErrorString(e));
    return 0;
}
CU
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  nvcc -O2 -arch=sm_${CC:-80} $T/t.cu -o $T/t 2>&1 | tail -5
  [ -x $T/t ] && $T/t
  if have ncu && [ -x $T/t ]; then
    echo; echo "-- ncu（普通用户）:"
    ncu --section SpeedOfLight -k saxpy -c 1 $T/t 2>&1 | grep -E "ERR|Duration|Throughput|Compute \(SM\)|Memory" | head -8
    echo "-- ncu（sudo -n）:"
    sudo -n env "PATH=$PATH" ncu --section SpeedOfLight -k saxpy -c 1 $T/t 2>&1 | grep -E "ERR|sudo|password|Duration|Throughput|Compute \(SM\)|Memory" | head -8
  fi
  rm -rf $T
else echo "没有 nvcc，跳过"; fi

sec "conda 环境 cuda"
bash -ic 'conda activate cuda 2>&1 && {
  echo "python: $(which python) $(python -V 2>&1)"
  echo "nvcc in env: $(which nvcc)"; nvcc --version 2>/dev/null | tail -1
  python - <<PY
try:
    import torch
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
    if torch.cuda.is_available(): print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
except Exception as e: print("torch 导入失败:", e)
for m in ("numpy","cutlass","cuda","triton"):
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "已安装"))
    except Exception as e: print(m, "未安装")
PY
}' 2>&1 | grep -v "no job control\|cannot set terminal process group\|Inappropriate ioctl"

sec "CUTLASS 与网络"
find / -maxdepth 6 -type d -name cutlass -path "*include*" 2>/dev/null | head -3
echo "-- 能否访问 GitHub（用于 clone CUTLASS）:"
timeout 15 git ls-remote --heads https://github.com/NVIDIA/cutlass 2>&1 | head -2
echo "-- 磁盘（家目录所在分区）:"; df -h ~ | tail -1
echo; echo "===== 体检结束 ====="
