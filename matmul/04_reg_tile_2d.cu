// matmul v4：二维寄存器分块 —— 每个线程算 C 里一个 TM×TN = 8×8 的小方块（64 个输出）。
// 思路（厨房比方）：厨师从货架取 8 份 A（一列）和 8 份 B（一行）拿在手里，两两搭配做 8×8 = 64 道菜。
//       取 16 次，做 64 次乘加；v3 是取 9 次做 8 次。
//
// 参数（命名规律：B* = 整个 block，T* = 单个线程；M = 行方向，N = 列方向，K = 被加掉的方向）
//   BM×BN = 128×128：一个 block 负责的 C 小块
//   BK = 8         ：沿 K 每步走几格
//   TM×TN = 8×8    ：一个线程负责的 C 小方块
//   线程数 = (BM/TM) × (BN/TN) = 16 × 16 = 256
//   每步要搬 A 小块 128×8 = 1024 个数、B 小块 8×128 = 1024 个数 → 每个线程各搬 4 个 A、4 个 B
#include "common.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NTHREADS ((BM / TM) * (BN / TN))   // 256

// ============================== 你要写的部分 ==============================
// 这一版同样假设 M % BM == 0、N % BN == 0、K % BK == 0（launch 里会检查）。
__global__ void gemm_reg2d(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];   // 128 × 8
    __shared__ float Bs[BK][BN];   // 8 × 128

    const int tx = threadIdx.x;              // 0..15：负责 C 小块的第 tx*TN .. tx*TN+7 列
    const int ty = threadIdx.y;              // 0..15：负责 C 小块的第 ty*TM .. ty*TM+7 行
    const int tid = ty * blockDim.x + tx;    // 0..255，用来分配搬运任务

    A += blockIdx.y * BM * K;                // 移到这个 block 负责的 BM 行
    B += blockIdx.x * BN;                    // 移到这个 block 负责的 BN 列
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM][TN] = {{0.0f}};   // 64 个累加器（寄存器）
    float regM[TM];                 // 这一个 kk 用到的 8 个 A（一列）
    float regN[TN];                 // 这一个 kk 用到的 8 个 B（一行）

    for (int t = 0; t < K; t += BK) {
        // TODO 1: 搬 A 小块（128×8 = 1024 个数，每个线程搬 4 个）。
        //   对 i = 0..3：idx = tid + i * NTHREADS；r = idx / BK；c = idx % BK；
        //               As[r][c] = A 第 r 行、第 (t + c) 列（行长 K）
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * NTHREADS;
            int r = idx / BK;
            int c = idx % BK;
            As[r][c] = A[r * K + t + c];
        }
        
        // TODO 2: 搬 B 小块（8×128 = 1024 个数，每个线程搬 4 个）。
        //   对 i = 0..3：idx = tid + i * NTHREADS；r = idx / BN；c = idx % BN；
        //               Bs[r][c] = B 第 (t + r) 行、第 c 列（行长 N）
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * NTHREADS;
            int r = idx / BN;
            int c = idx % BN;
            Bs[r][c] = B[(t + r) * N + c];
        }

        // TODO 3: 同步
        __syncthreads();

        // TODO 4: 对 kk = 0..BK-1：
        //   (a) 把 8 个 A 读进寄存器：regM[i] = As[ty*TM + i][kk]，i = 0..TM-1
        //   (b) 把 8 个 B 读进寄存器：regN[j] = Bs[kk][tx*TN + j]，j = 0..TN-1
        //   (c) 两两相乘累加（外积）：acc[i][j] += regM[i] * regN[j]
        for (int kk = 0; kk < BK; ++kk) {
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[ty * TM + i][kk];
            }
            for (int j = 0; j < TN; ++j) {
                regN[j] = Bs[kk][tx * TN + j];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regM[i] * regN[j];
                }
            }
        }

        // TODO 5: 同步
        __syncthreads();
    }

    // TODO 6: 写回 64 个结果：C 小块第 (ty*TM + i) 行、第 (tx*TN + j) 列 = acc[i][j]（行长 N）
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            C[(ty * TM + i) * N + (tx * TN + j)] = acc[i][j];
        }
    }
}
// ========================================================================

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "04_reg_tile_2d 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK);
        exit(1);
    }
    dim3 block(BN / TN, BM / TM);   // (16, 16) = 256 个线程
    dim3 grid(N / BN, M / BM);
    gemm_reg2d<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    return run_gemm_benchmark("04_reg_tile_2d", launch, argc, argv);
}
