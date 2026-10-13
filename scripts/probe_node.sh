#!/bin/bash
# 在 GPU 节点上运行的探针（由 scripts/probe.sh 触发），工作目录 = 远端项目根目录
echo "主机: $(hostname)   用户: $(whoami)   uid/gid: $(id -u)/$(id -g)   umask: $(umask)"
echo "工作目录: $PWD"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<无>}   进程/线程数: $(ps -u "$(whoami)" -L --no-headers 2>/dev/null | wc -l) / ulimit -u $(ulimit -u)"
echo "--- 目录权限 ---"
ls -ld . logs results scripts 2>&1
echo "--- 能否在 logs/ 里新建文件 ---"
f="logs/probe_$$.txt"
if (echo ok > "$f") 2>&1; then echo "[ OK ] 写入成功"; rm -f "$f"; else echo "[FAIL] 无法写入 logs/"; fi
echo "--- 磁盘 / 配额 ---"
df -h . 2>&1 | tail -1
quota -s 2>&1 | sed -n 1,6p
echo "--- 挂载 ---"
mount 2>/dev/null | grep -E "research|uac" | sed -n 1,4p
echo "--- 加载 CUDA 环境 ---"
if source scripts/cse_gpu_env.sh 2>&1; then which nvcc && nvcc --version | tail -1; fi
echo "--- GPU ---"
nvidia-smi -L 2>&1
