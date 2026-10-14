"""参照线 2：PyTorch 的 F.layer_norm 耗时 —— 用与 common.cuh 相同的方法测。
  热 L2 / 冷 L2（每次计时前写一块 256 MB 缓冲区把 L2 冲掉），热身 10 次，计时 100 次取中位数。
  eager 会把 PyTorch 每次调用的 CPU 开销也量进去；cuda graph 把它去掉，更接近 kernel 本身。
用法:  CUDA_VISIBLE_DEVICES=0 python ref/bench_torch.py        （N、H 取自 data/meta.txt）
环境变量: PEAK_GBPS（默认 1597）  FLUSH_MB（清 L2 的缓冲区大小，默认 256）
"""
import os

import torch
import torch.nn.functional as F


def read_meta():
    here = os.path.dirname(os.path.abspath(__file__))
    n, h, eps = open(os.path.join(here, "..", "data", "meta.txt")).read().split()
    return int(n), int(h), float(eps)


def time_it(fn, flush_buf, iters=100, warmup=10):
    """只围住 fn() 计时；flush_buf 不为 None 时，每次计时前先清 L2（不计入耗时）。返回 (中位数 ms, 最小值 ms)。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if flush_buf is not None:
            flush_buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[iters // 2], times[0]


def main():
    N, H, eps = read_meta()
    peak = float(os.environ.get("PEAK_GBPS", 1597))
    flush_mb = int(os.environ.get("FLUSH_MB", 256))
    dev = torch.device("cuda")
    print("== torch F.layer_norm ==")
    print(f"Device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}   N={N} H={H} eps={eps}")

    x = torch.randn(N, H, device=dev, dtype=torch.float32)
    gamma = torch.randn(H, device=dev, dtype=torch.float32)
    beta = torch.randn(H, device=dev, dtype=torch.float32)
    eager = lambda: F.layer_norm(x, (H,), gamma, beta, eps)

    # CUDA graph：把一次调用录下来再回放，去掉每次调用的 CPU 开销
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            eager()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eager()
    replay = graph.replay

    flush_buf = torch.empty(flush_mb << 20, dtype=torch.uint8, device=dev)
    nbytes = 2.0 * N * H * 4  # 最少搬运：x 读一次 + y 写一次

    def gbps(ms):
        return nbytes / (ms * 1e6)

    for mode, fn in (("eager", eager), ("cuda graph", replay)):
        for cache, fb in (("热 L2", None), ("冷 L2", flush_buf)):
            med, mn = time_it(fn, fb)
            print(f"{mode:10s} {cache}:  median = {med:.3f} ms ({gbps(med):7.1f} GB/s, {100 * gbps(med) / peak:5.1f}% of {peak:.0f})"
                  f"   min = {mn:.3f} ms")


if __name__ == "__main__":
    main()
