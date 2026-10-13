"""在本机生成测试数据：输入 x、gamma、beta，以及 PyTorch 算出的参考输出 y_ref。
用法:  python ref/gen_data.py [N] [H]      默认 N=4096, H=1024
产物:  data/{x,gamma,beta,y_ref}.bin（裸 float32）和 data/meta.txt（"N H eps"）
"""
import os
import sys

import numpy as np
import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
EPS = 1e-5

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(out, exist_ok=True)

rng = np.random.default_rng(42)
x = rng.standard_normal((N, H)).astype(np.float32)
gamma = rng.standard_normal(H).astype(np.float32)   # 故意不用全 1，才能暴露 gamma 下标写错
beta = rng.standard_normal(H).astype(np.float32)    # 同理，beta 也不用全 0

y = torch.nn.functional.layer_norm(
    torch.from_numpy(x), (H,), torch.from_numpy(gamma), torch.from_numpy(beta), EPS
).numpy()

for name, arr in [("x", x), ("gamma", gamma), ("beta", beta), ("y_ref", y)]:
    arr.astype(np.float32).tofile(os.path.join(out, f"{name}.bin"))
with open(os.path.join(out, "meta.txt"), "w") as f:
    f.write(f"{N} {H} {EPS}\n")

print(f"已生成 data/: N={N} H={H} eps={EPS}  x 约 {x.nbytes / 2**20:.0f} MB")
print("下一步: scripts/push_data.sh")
