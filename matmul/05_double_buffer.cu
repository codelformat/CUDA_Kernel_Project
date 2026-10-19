// matmul v5：双缓冲（预取）—— 算这一块的同时，把下一块从显存读进来。
// 在 v4 的基础上只改一件事：
//   v4 每步：读显存 → 写共享内存 → 同步 → 算 → 同步（读显存时计算单元空等）
//   v5 每步：发出"下一块"的显存读（结果先放寄存器，数据在路上）
//            → 用"这一块"计算（显存读在后台进行）
//            → 把寄存器里的下一块写进另一份共享内存 → 同步一次 → 两份缓冲互换
// 共享内存开两份：As[2]、Bs[2]，一份在被读（算），另一份在被写（搬）。
#include "common.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NTHREADS ((BM / TM) * (BN / TN))   // 256
#define LOADS_A (BM * BK / NTHREADS)        // 每个线程每步搬几个 A：4
#define LOADS_B (BK * BN / NTHREADS)        // 每个线程每步搬几个 B：4

// ============================== 你要写的部分 ==============================
__global__ void gemm_db(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[2][BM][BK];   // 两份：[0] 和 [1]
    __shared__ float Bs[2][BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM][TN] = {{0.0f}};
    float regM[TM], regN[TN];
    float ldA[LOADS_A], ldB[LOADS_B];   // 下一块的数据先暂存在寄存器里

    // ---- 开场：把第 0 块搬进第 0 份缓冲（和 v4 的 TODO 1、2 一样，只是写进 As[0]、Bs[0]）----
    // TODO 1: 对 i = 0..LOADS_A-1：idx = tid + i*NTHREADS; r = idx/BK; c = idx%BK; As[0][r][c] = A[r*K + c]
    //         对 i = 0..LOADS_B-1：idx = tid + i*NTHREADS; r = idx/BN; c = idx%BN; Bs[0][r][c] = B[r*N + c]

    __syncthreads();

    int cur = 0;                       // 这一步要读（计算）的是第 cur 份
    for (int t = 0; t < K; t += BK) {
        const bool has_next = (t + BK < K);

        // TODO 2: 如果 has_next：发出下一块（起点 t + BK）的显存读，结果只放进 ldA[i]、ldB[i]，先不要写共享内存。
        //         A：ldA[i] = A 第 r 行、第 (t + BK + c) 列；B：ldB[i] = B 第 (t + BK + r) 行、第 c 列
        //         （r、c 的算法和 TODO 1 一样）

        // TODO 3: 用 As[cur]、Bs[cur] 计算这一块：和 v4 的 TODO 4 完全一样，只是数组多了一维 [cur]

        // TODO 4: 如果 has_next：把 ldA[i]、ldB[i] 写进另一份缓冲 As[cur ^ 1]、Bs[cur ^ 1] 的对应位置
        //         （cur ^ 1 是"异或 1"：0 变 1，1 变 0）

        // TODO 5: 同步一次，然后切换：cur ^= 1
    }

    // TODO 6: 写回 64 个结果（和 v4 一样）
}
// ========================================================================

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "05_double_buffer 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK);
        exit(1);
    }
    dim3 block(BN / TN, BM / TM);
    dim3 grid(N / BN, M / BM);
    gemm_db<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    return run_gemm_benchmark("05_double_buffer", launch, argc, argv);
}
