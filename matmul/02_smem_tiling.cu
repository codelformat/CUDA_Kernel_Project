// matmul v2：共享内存分块（tiling）
// 思路：一个 block 负责 C 的一个 TILE×TILE 小块。沿 K 方向每次把 A 的一个 TILE×TILE 小块和 B 的一个 TILE×TILE 小块
//       搬进共享内存（每个线程只搬 1 个 A 元素 + 1 个 B 元素），然后整个 block 在共享内存里反复使用它们。
// 每个线程仍然只算 C 的一个元素。
#include "common.cuh"

#ifndef TILE
#define TILE 32          // 线程块是 TILE×TILE 个线程；TILE=32 时为 1024 个线程
#endif

// ============================== 你要写的部分 ==============================
__global__ void gemm_smem(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];   // 当前这一步的 A 小块：A[blockIdx.y*TILE .. +TILE][t .. t+TILE]
    __shared__ float Bs[TILE][TILE];   // 当前这一步的 B 小块：B[t .. t+TILE][blockIdx.x*TILE .. +TILE]

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;   // 这个线程负责的 C 的行（x 对应列，沿用第 2 课的结论）
    const int col = blockIdx.x * TILE + tx;   // 这个线程负责的 C 的列

    // 注意：这里不能像 v1 那样一开始就对越界线程 return。想一想为什么（提示：__syncthreads）。
    float sum = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        // TODO 1: 把 A[row][t + tx] 搬到 As[ty][tx]；越界（row >= M 或 t + tx >= K）就填 0。

        // TODO 2: 把 B[t + ty][col] 搬到 Bs[ty][tx]；越界（t + ty >= K 或 col >= N）就填 0。

        // TODO 3: 等 block 里所有线程都搬完。

        // TODO 4: 用共享内存里的两个小块累加：kk 从 0 到 TILE-1，sum += A 小块第 ty 行第 kk 个 × B 小块第 kk 行第 tx 个。

        // TODO 5: 再等一次所有线程。（为什么需要这第二次同步？）
    }

    // TODO 6: 只有不越界的线程写回 C[row][col]。
}
// ========================================================================

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid(CDIV(N, TILE), CDIV(M, TILE));
    gemm_smem<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    return run_gemm_benchmark("02_smem_tiling", launch, argc, argv);
}
