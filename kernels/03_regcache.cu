// 第 9 课：v3 —— x 只从全局内存读一次，缓存在寄存器里。
// 相对 v2 只改一件事：v2 的三遍扫描每遍都重新去读 x；v3 在第一遍把自己负责的元素读进寄存器，后两遍直接用。
// 其余不变：一个 block 一行，共享内存树形归约，仍然是均值 -> 方差 -> 输出三步。
#include "common.cuh"
#include "block_reduce.cuh"   // block_reduce_sum（就是你在 v2 里写的那个）

// ITEMS = 每个 thread 负责的元素个数 = H / blockDim.x。
// 它必须是【编译期常量】（模板参数）：这样 v[ITEMS] 的每个下标编译时都已知，数组才会被放进寄存器；
// 如果下标要到运行时才知道，编译器会把数组放进"本地内存"(local memory，物理上在显存里)，等于白忙一场。
template <int ITEMS>
__global__ void layernorm_regs(const float* x, const float* gamma, const float* beta,
                               float* y, int N, int H, float eps) {
    __shared__ float smem[MAX_THREADS];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * H;
    float* yr = y + (size_t)row * H;
    (void)N;

    float v[ITEMS];   // 这个 thread 的 ITEMS 个元素，常驻寄存器

    // ============================== 你要写的部分 ==============================
    // TODO 1: 第一遍。对 i = 0 .. ITEMS-1：读 v[i] = xr[tid + i * blockDim.x]，同时累加部分和；
    //         然后 block_reduce_sum -> 除以 H 得到 mean。
    //         循环前加 #pragma unroll，且循环上界必须写 ITEMS（常量），不能写 H / blockDim.x。
    float mean = 0.f;

    // TODO 2: 第二遍。这次【不再碰 xr】：直接用 v[i] 累加 (v[i] - mean)^2 的部分和 -> 归约 -> 除以 H 得到 var。
    float var = 0.f;

    // TODO 3: 第三遍。同样只用 v[i]：
    //         yr[tid + i * blockDim.x] = (v[i] - mean) / sqrtf(var + eps) * gamma[该元素的列号] + beta[该元素的列号]。
    //         （gamma、beta 每次还是从全局内存读，它们很小，会一直待在缓存里，这一版先不管。）
    // ========================================================================
    (void)v; (void)mean; (void)var; (void)yr; (void)gamma; (void)beta; (void)eps;
}

void launch(const float* x, const float* gamma, const float* beta, float* y,
            int N, int H, float eps, int threads) {
    if (threads < 32 || threads > MAX_THREADS || (threads & (threads - 1)) != 0 || H % threads != 0) {
        fprintf(stderr, "threads 必须是 2 的幂、在 [32, %d] 之间，且能整除 H=%d，现在是 %d\n", MAX_THREADS, H, threads);
        exit(1);
    }
    // 模板参数必须是编译期常量，所以运行时用 switch 挑一个实例化好的版本
    switch (H / threads) {
        case 1:  layernorm_regs<1><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        case 2:  layernorm_regs<2><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        case 4:  layernorm_regs<4><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        case 8:  layernorm_regs<8><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        case 16: layernorm_regs<16><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        default:
            fprintf(stderr, "H / threads = %d，不在支持的 {1, 2, 4, 8, 16} 里\n", H / threads);
            exit(1);
    }
    CK(cudaGetLastError());
}

int main(int argc, char** argv) {
    return run_benchmark("03_regcache", launch, argc, argv);
}
