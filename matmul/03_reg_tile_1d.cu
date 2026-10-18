// matmul v3：一维寄存器分块 —— 每个线程算 C 同一行里连续 TN 个元素。
// 思路：v2 每做 1 次乘加就要从共享内存读 1 个 A、1 个 B。
//       现在一个线程负责 TN 个输出：从共享内存读 1 个 A 放进寄存器，和 TN 个 B 各乘一次。
//       这个 A 在寄存器里被复用 TN 次，"共享内存读 / FFMA"随之下降。
//
// 分块参数：block 负责 C 的 BM×BN 小块；沿 K 每步处理 BK 个 k。
//   线程数 = BM × (BN / TN) = 64 × 8 = 512；线程 (ty, tx) 负责 C 小块第 ty 行、第 tx*TN .. tx*TN+TN-1 列。
//   每步要搬 A 小块 BM×BK = 512 个数、B 小块 BK×BN = 512 个数 —— 正好每个线程各搬 1 个 A、1 个 B。
#include "common.cuh"

#define BM 64
#define BN 64
#define BK 8
#define TN 8
#define NTHREADS (BM * (BN / TN))   // 512

// ============================== 你要写的部分 ==============================
// 为了简单，这一版假设 M % BM == 0、N % BN == 0、K % BK == 0（launch 里会检查）。
__global__ void gemm_reg1d(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];   // A 小块：A[blockIdx.y*BM .. +BM][t .. t+BK]
    __shared__ float Bs[BK][BN];   // B 小块：B[t .. t+BK][blockIdx.x*BN .. +BN]

    const int tx = threadIdx.x;              // 0 .. BN/TN-1（= 0..7）
    const int ty = threadIdx.y;              // 0 .. BM-1   （= 0..63）
    const int tid = ty * blockDim.x + tx;    // 线性序号 0..511，用来分配搬运任务

    // 搬运分工（和计算分工不同！搬运时按 tid 重新排，让一个 warp 读连续地址）：
    //   A 小块是 64 行 × 8 列：tid 负责第 tid / BK 行、第 tid % BK 列
    //   B 小块是 8 行 × 64 列：tid 负责第 tid / BN 行、第 tid % BN 列
    const int aRow = tid / BK, aCol = tid % BK;
    const int bRow = tid / BN, bCol = tid % BN;

    // 移动指针到这个 block 负责的起点，后面的下标都相对它
    A += blockIdx.y * BM * K;   // A 小块所在的 BM 行
    B += blockIdx.x * BN;       // B 小块所在的 BN 列
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TN] = {0.0f};     // TN 个累加器，编译器会把它放进寄存器（下标必须是编译期可知的）

    for (int t = 0; t < K; t += BK) {
        // TODO 1: 搬 A：As[aRow][aCol] = A 小块里第 aRow 行、第 (t + aCol) 列的元素（用移动后的 A 指针，行长是 K）
        As[aRow][aCol] = A[aRow * K + t + aCol];

        // TODO 2: 搬 B：Bs[bRow][bCol] = B 第 (t + bRow) 行、第 bCol 列的元素（用移动后的 B 指针，行长是 N）
        Bs[bRow][bCol] = B[(t + bRow) * N + bCol];

        // TODO 3: 同步
        __syncthreads();

        // TODO 4: 对 kk = 0..BK-1：
        //           先把 As[ty][kk] 读进一个寄存器变量 a，
        //           再对 j = 0..TN-1：acc[j] += a * Bs[kk][tx * TN + j]
        for (int kk = 0; kk < BK; ++kk) {
            float a = As[ty][kk];
            for (int j = 0; j < TN; ++j) {
                acc[j] += a * Bs[kk][tx * TN + j];
            }
        }

        // TODO 5: 同步
        __syncthreads();
    }

    // TODO 6: 写回 TN 个结果：C 小块第 ty 行、第 tx*TN + j 列（j = 0..TN-1），行长是 N
    for (int j = 0; j < TN; ++j) {
        C[ty * N + tx * TN + j] = acc[j];
    }
}
// ========================================================================

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "03_reg_tile_1d 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK);
        exit(1);
    }
    dim3 block(BN / TN, BM);    // (8, 64) = 512 个线程
    dim3 grid(N / BN, M / BM);
    gemm_reg1d<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    return run_gemm_benchmark("03_reg_tile_1d", launch, argc, argv);
}
