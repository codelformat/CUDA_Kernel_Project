#!/usr/bin/env bash
# 诊断作业: scripts/why.sh <jobid>
# scontrol 只保留作业结束后约 5 分钟的信息，所以失败后尽快运行；run.sh 在作业失败时会自动调用本脚本。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/config.sh"
ID="${1:?用法: scripts/why.sh <jobid>}"
require_master || exit 1

rsh <<EOF
$(slurm_pre)
echo "--- scontrol show job ${ID} ---"
out=\$(scontrol show job ${ID} 2>&1)
if echo "\$out" | grep -q JobState; then
  echo "\$out" | tr ' ' '\n' | grep -E '^(JobState|Reason|ExitCode|DerivedExitCode|NodeList|RunTime|SubmitTime|StartTime|EndTime|StdOut|StdErr|WorkDir)='
else
  echo "\$out" | sed -n 1,2p
fi
echo "--- sacct ---"
sacct -j ${ID} --format=JobID,State,ExitCode,Elapsed,NodeList 2>&1 | sed -n 1,6p
echo "--- 远端项目目录 ---"
if cd $(printf %q "$REMOTE_DIR") 2>/dev/null; then
  pwd; ls -la logs 2>&1 | tail -4; ls -la scripts/job.sbatch scripts/cse_gpu_env.sh 2>&1
  df -h . 2>/dev/null | tail -1
else
  echo "进不去 ${REMOTE_DIR}"
fi
echo "--- 登录节点上你的资源占用 ---"
echo "进程/线程数: \$(ps -u \$(whoami) -L --no-headers 2>/dev/null | wc -l)   (ulimit -u = \$(ulimit -u))"
echo "带终端的登录数: \$(who | awk -v u=\$(whoami) '\$1==u' | wc -l)   (上限 5)"
EOF
