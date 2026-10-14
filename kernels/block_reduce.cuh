// 块内求和归约（共享内存 + 树形）。就是 v2 里写的 block_reduce_sum，抽出来给后面的版本复用。
// 要求 blockDim.x 是 2 的幂，且 <= MAX_THREADS；smem 至少有 blockDim.x 个 float。
#pragma once

#define MAX_THREADS 1024   // 一个 block 最多 1024 个 thread

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();                       // 所有人都写完，才允许开始读别人的
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();                   // 这一轮都写完，才进入下一轮（必须在 if 外面）
    }
    const float total = smem[0];
    __syncthreads();                       // 所有人都读完 smem[0]，下一次调用才可以覆盖 smem
    return total;
}
