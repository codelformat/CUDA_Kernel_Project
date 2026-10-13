"""第 1 课作业：用 NumPy 亲手实现 LayerNorm（forward），并和 PyTorch 对拍。

规则：
  - 只能用 numpy 的基础运算（mean / sum / sqrt 等都可以），不能调用任何现成的 layernorm。
  - 输入 x 形状 [N, H]：N 行（= batch * seq_len 个 token），每行 H 个特征。
  - 对每一行单独归一化，然后逐元素乘 gamma、加 beta（gamma、beta 形状都是 [H]）。
"""
import numpy as np


def layernorm_np(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    # TODO(你来写): 按公式实现，返回形状 [N, H]、dtype=float32 的数组
    x_mean = np.mean(x, axis=1, keepdims=True)  # 计算每行的均值
    # x_var = np.var(x, axis=1, keepdims=True)  # 计算每行的方差
    # 手写计算方差
    H = x.shape[-1]
    x_var = np.sum((x - x_mean) ** 2, axis=1, keepdims=True) / H
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)  # 标准化
    y = x_normalized * gamma + beta  # 逐元素乘 gamma，加 beta
    return y.astype(np.float32)  # 确保输出为 float32


def main():
    import torch  # 没有的话先运行: scripts/setup_env.sh ; source .venv/bin/activate

    rng = np.random.default_rng(0)
    ok = True
    for N, H in [(4, 8), (128, 1024), (4096, 768)]:
        x = (rng.standard_normal((N, H)) * 3 + 2).astype(np.float32)
        gamma = rng.standard_normal(H).astype(np.float32)
        beta = rng.standard_normal(H).astype(np.float32)

        y = layernorm_np(x, gamma, beta)
        ref = torch.nn.functional.layer_norm(
            torch.from_numpy(x), (H,), torch.from_numpy(gamma), torch.from_numpy(beta), 1e-5
        ).numpy()

        assert y.shape == ref.shape, f"形状不对: {y.shape} vs {ref.shape}"
        err = float(np.abs(y - ref).max())
        status = "PASS" if err < 1e-5 else "FAIL"
        ok &= err < 1e-5
        print(f"N={N:5d} H={H:5d}  max_abs_err={err:.3e}  {status}")

    # import numpy as np
    x = np.full((1, 8), 3.0, dtype=np.float32)
    g, b = np.ones(8, np.float32), np.zeros(8, np.float32)
    print(layernorm_np(x, g, b, eps=1e-5))
    print(layernorm_np(x, g, b, eps=0.0))
    
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
