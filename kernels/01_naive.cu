// v1：最朴素的 LayerNorm kernel —— 一个 thread 负责一整行，三遍扫描（均值 -> 方差 -> 输出）。
// 目标只有一个：结果正确。性能是后续各版本要改进的基线。
// 测试框架（读数据 / 正确性校验 / 计时 / 带宽）与其他版本共用 common.cuh，
// 因此可以用 FLUSH_L2=1 在"冷 L2"下和 v2~v4 做同口径对比。
#include "common.cuh"

// 数据是 row-major：第 r 行第 j 个元素在 x[r * H + j]。
__global__ void layernorm_naive(const float* x, const float* gamma, const float* beta,
                                float* y, int N, int H, float eps) {
    // 这个 thread 负责哪一行
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // 越界保护：grid 是向上取整算出来的，N 不一定是 blockDim 的整数倍
    if (row >= N) return;
    int offset = row * H;
    // 第一遍扫描这一行，求均值
    float mean = 0;
    for (int j = 0; j < H; ++j) {
        mean += x[offset + j];
    }
    mean /= H;
    // 第二遍扫描这一行，求方差（除以 H）
    float var = 0;
    for (int j = 0; j < H; ++j) {
        float diff = x[offset + j] - mean;
        var += diff * diff;
    }
    var /= H;
    // 第三遍扫描这一行，算出 y 写回去
    for (int j = 0; j < H; ++j) {
        y[offset + j] = (x[offset + j] - mean) / sqrt(var + eps) * gamma[j] + beta[j];
    }
}

void launch(const float* x, const float* gamma, const float* beta, float* y,
            int N, int H, float eps, int threads) {
    const int blocks = (N + threads - 1) / threads;
    layernorm_naive<<<blocks, threads>>>(x, gamma, beta, y, N, H, eps);
    CK(cudaGetLastError());
}

int main(int argc, char** argv) {
    return run_benchmark("01_naive", launch, argc, argv);
}
