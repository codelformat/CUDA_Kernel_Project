#!/usr/bin/env bash
# 用 Nsight Compute 抓一组常用指标，整理成一张表（不重复跑 --set full，几秒钟出结果）。
# 用法:   scripts/ncu_quick.sh <可执行文件> <kernel 名正则> [程序参数...]
# 例:     scripts/ncu_quick.sh results/01_naive_20261017-090000/01_naive gemm_naive 1024 1024 1024
# 环境变量: NCU_METRICS="a,b,c"  覆盖默认指标      NCU_COUNT=1  抓取前几次匹配的 kernel 启动
# 默认用 --clock-control none：GPU 按真实时钟运行，耗时与平时一致（ncu 默认会锁到基础频率）。
set -eo pipefail
[ $# -ge 2 ] || { sed -n '2,7p' "$0"; exit 2; }
BIN="$1"; KRE="$2"; shift 2
[ -x "$BIN" ] || { echo "找不到可执行文件 $BIN"; exit 2; }
command -v ncu >/dev/null 2>&1 || { echo "PATH 里没有 ncu"; exit 1; }
DEFAULT_METRICS="gpu__time_duration.sum,smsp__cycles_elapsed.avg.per_second,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed_op_shared_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
METRICS="${NCU_METRICS:-$DEFAULT_METRICS}"
RAW=$(mktemp); trap 'rm -f "$RAW"' EXIT
ncu --csv --clock-control none -k "regex:$KRE" -c "${NCU_COUNT:-1}" --metrics "$METRICS" "$BIN" "$@" >"$RAW" 2>&1 || true
python3 -I - "$RAW" <<'PY'
import csv, sys
rows, errs = [], []
for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
    if line.startswith('"'):
        rows.append(line)
    elif 'ERR' in line or 'Failed' in line or 'rror' in line:
        errs.append(line.rstrip())
if not rows:
    print("没有抓到数据。ncu 的输出里与错误相关的行:")
    print("\n".join(errs[:10]) or "(无)")
    sys.exit(1)
vals, kernel = {}, ""
for r in csv.DictReader(rows):
    kernel = r.get("Kernel Name", kernel)
    v = r["Metric Value"].replace(",", "")
    try: v = float(v)
    except ValueError: pass
    vals[r["Metric Name"]] = (v, r["Metric Unit"])
print(f"kernel: {kernel}")
for name, (v, unit) in vals.items():
    print(f"  {name:62s} {v:>16,.2f} {unit}" if isinstance(v, float) else f"  {name:62s} {v:>16} {unit}")
def g(n): return vals.get(n, (None,))[0]
for kind, tag in (("ld", "load"), ("st", "store")):
    req = g(f"l1tex__t_requests_pipe_lsu_mem_global_op_{kind}.sum")
    sec = g(f"l1tex__t_sectors_pipe_lsu_mem_global_op_{kind}.sum")
    if req and sec: print(f"\n>> 全局 {tag}：每个 request 平均 {sec / req:.2f} 个 sector（连续 4 B/线程 = 4；全部同地址 = 1；各踩一条线 = 32）")
ffma = g("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
gld = g("smsp__inst_executed_op_global_ld.sum"); sld = g("smsp__inst_executed_op_shared_ld.sum")
if ffma:
    wf = ffma / 32
    print(f">> warp 级 FFMA ≈ {wf:,.0f} 条；每条 FFMA 配 {(gld or 0) / wf:.3f} 条全局读、{(sld or 0) / wf:.3f} 条共享内存读")
clk = vals.get("smsp__cycles_elapsed.avg.per_second")
if clk:
    ghz = clk[0] if "nsecond" in clk[1] else clk[0] / 1e9   # 单位可能是 cycle/nsecond（即 GHz）或 cycle/second
    print(f">> SM 时钟 ≈ {ghz:.2f} GHz")
PY
