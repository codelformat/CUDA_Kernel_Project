// 第 5 课：v2 —— 一行交给一个 block 协作处理。
// 本版只改一件事：并行方式（一个 thread 一行 -> 一个 block 一行）。
// 算法不变：仍然是三遍扫描（均值 -> 方差 -> 输出），每遍读全局内存。
#include "common.cuh"

#define MAX_THREADS 1024   // 一个 block 最多 1024 个 thread

// ============================== 你要写的部分 ==============================
// 块内求和归约：每个 thread 带着自己的部分和 val 进来，所有 thread 都拿到整个 block 的总和。
// 要求 blockDim.x 是 2 的幂（且 <= MAX_THREADS）。smem 是这个 block 共享的一块片上存储。
__device__ float block_reduce_sum(float val, float* smem) {
    const int tid = threadIdx.x;
    // TODO 1: 把自己的 val 写进 smem[tid]，然后 __syncthreads()。
    //         想一想：不同步的话，别的 thread 读 smem[tid + s] 时，那里可能还是什么？
    smem[tid] = val;
    __syncthreads();
    // TODO 2: 树形归约。步长 s 从 blockDim.x / 2 开始，每轮减半，直到 s == 1：
    //         每轮让 tid < s 的 thread 执行 smem[tid] += smem[tid + s]；每轮结束都 __syncthreads()。
    //         注意：__syncthreads() 必须放在 if 外面——block 里所有 thread 都得执行到它，否则死锁/未定义行为。
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    // TODO 3: 总和在 smem[0]。读出来存进 total，然后再 __syncthreads() 一次：
    //         因为这个函数会被连续调用两次，下一次会重新写 smem，必须等所有 thread 都读完 smem[0]。
    float total = smem[0];
    __syncthreads();
    return total;
}

__global__ void layernorm_block(const float* x, const float* gamma, const float* beta,
                                float* y, int N, int H, float eps) {
    __shared__ float smem[MAX_THREADS];
    const int tid = threadIdx.x;
    // TODO 4: 这个 block 负责哪一行？（一个 block 一行，所以 grid 里有 N 个 block）
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * H;   // 这一行的起点
    float* yr = y + (size_t)row * H;

    // TODO 5: 第一遍求均值。每个 thread 以 blockDim.x 为步长跨着读：j = tid, tid + blockDim.x, ...（j < H），
    //         累加成自己的部分和；再调用 block_reduce_sum 得到整行总和，除以 H 得到 mean。
    //         （这种读法让一个 warp 的 32 个 thread 读到连续的 128 B —— 这就是合并访存。）
    float mean = 0.f;
    for (int j = tid; j < H; j += blockDim.x) {
        mean += xr[j];
    }
    mean = block_reduce_sum(mean, smem) / H;

    // TODO 6: 第二遍求方差。同样的循环，累加 (x[j] - mean)^2 的部分和 -> 归约 -> 除以 H 得到 var。
    float var = 0.f;
    for (int j = tid; j < H; j += blockDim.x) {
        float diff = xr[j] - mean;
        var += diff * diff;
    }
    var = block_reduce_sum(var, smem) / H;

    // TODO 7: 第三遍写输出。同样的循环：yr[j] = (xr[j] - mean) / sqrt(var + eps) * gamma[j] + beta[j]。
    for (int j = tid; j < H; j += blockDim.x) {
        yr[j] = (xr[j] - mean) / sqrtf(var + eps) * gamma[j] + beta[j];
    }
}

void launch(const float* x, const float* gamma, const float* beta, float* y,
            int N, int H, float eps, int threads) {
    if (threads < 32 || threads > MAX_THREADS || (threads & (threads - 1)) != 0) {
        fprintf(stderr, "threads 必须是 2 的幂，且在 [32, %d] 之间，现在是 %d\n", MAX_THREADS, threads);
        exit(1);
    }
    // TODO 8: 启动 kernel：N 个 block，每个 block threads 个 thread（不需要动态共享内存）。
    layernorm_block<<<N, threads>>>(x, gamma, beta, y, N, H, eps);
    CK(cudaGetLastError());
}
// ========================================================================

int main(int argc, char** argv) {
    return run_benchmark("02_block_per_row", launch, argc, argv);
}
