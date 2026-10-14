// 参照线 1：纯拷贝 kernel（y = x）的带宽。
// 它在"读一遍 x + 写一遍 y"这个访存模式下，量出这块卡实际能跑到多少 GB/s —— 不做任何计算，
// 所以是 LayerNorm 在同样数据量下的天花板。用 float4 向量化 + 合并访问，写法尽量贴近硬件极限。
// 注意：要用 FLUSH_L2=1 才是真正的显存带宽。
#include "common.cuh"

__global__ void copy_f4(const float4* __restrict__ x, float4* __restrict__ y, size_t n4) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        y[i] = x[i];
    }
}

void launch(const float* x, const float* gamma, const float* beta, float* y,
            int N, int H, float eps, int threads) {
    (void)gamma; (void)beta; (void)eps;
    const size_t n = (size_t)N * H;
    if (n % 4 != 0) { fprintf(stderr, "N*H 必须是 4 的倍数\n"); exit(1); }
    const size_t n4 = n / 4;
    const int blocks = (int)((n4 + threads - 1) / threads);
    copy_f4<<<blocks, threads>>>(reinterpret_cast<const float4*>(x), reinterpret_cast<float4*>(y), n4);
    CK(cudaGetLastError());
}

int main(int argc, char** argv) {
    return run_benchmark("bench_copy (y = x, float4)", launch, argc, argv, /*check=*/false);
}
