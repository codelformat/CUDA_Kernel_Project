// matmul 阶段 B · tc02：共享内存分块 + 每个 warp 算多个 16×16 片段（WMMA）
// 对应阶段 A 的 v2（共享内存）+ v4（一个线程算多个输出）+ v8（warp 级分块），这次一步到位，因为这些道理你都学过了。
//
// tc01 的问题：每个 warp 自己从显存读 16×16 的 A、B，每个数只被用 16 次，4096³ 要搬 16 GB。
// tc02：
//   block 负责 C 的 128×128，每步沿 K 走 BK=32。
//   ① block 的 256 个线程合作把 A 的 128×32、B 的 32×128 搬进共享内存（每个数从显存只读一次，在片上被 8 个 warp 共用）。
//   ② 8 个 warp 排成 2 行 × 4 列，每个 warp 负责 C 的 64×32 = 4 × 2 个 16×16 片段。
//      每走 16 个 k：读 4 个 A 片段（4 行）、2 个 B 片段（2 列），做 4 × 2 = 8 次 mma —— 一个 A 片段和 2 个 B 片段相乘，一个 B 片段和 4 个 A 片段相乘（外积，寄存器层面的复用）。
//   这一版没有双缓冲、没有 padding，一次只学一件事；bank 冲突留到后面量。
//
// 数据类型：half 是 2 字节，搬运时用 uint4（16 字节 = 8 个 half）一次搬 8 个。
#include "common_tc.cuh"
#include <mma.h>
using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 32
#define NTHREADS 256
#define WM 16
#define WN 16
#define WK 16
#define WTM 64               // warp 负责的行数
#define WTN 32               // warp 负责的列数
#define WARPS_X (BN / WTN)   // 4：一行有几个 warp
#define FM (WTM / WM)        // 4：warp 里竖着几个片段
#define FN (WTN / WN)        // 2：warp 里横着几个片段

__global__ void __launch_bounds__(NTHREADS) gemm_wmma_smem(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ __align__(128) half As[BM][BK];   // A 的 128×32 小块（不转置：load_matrix_sync 自己会按行主序读）
    __shared__ __align__(128) half Bs[BK][BN];   // B 的 32×128 小块

    const int tid = threadIdx.x;
    A += blockIdx.y * BM * K;                    // 移到这个 block 负责的 A 的第一行
    B += blockIdx.x * BN;                        // 移到这个 block 负责的 B 的第一列
    C += blockIdx.y * BM * N + blockIdx.x * BN;  // 移到这个 block 负责的 C 的左上角

    // ============================== 你要写的部分（一）==============================
    // TODO 1: 这个 warp 在 block 里负责哪块 64×32？（同 v8：8 个 warp 排成 2 行 × 4 列）
    //         算出 warpRow0（0 或 64）、warpCol0（0、32、64、96）
    int warpRow0 = 0, warpCol0 = 0;
    // ===============================================================================

    // 8 个累加片段：acc[i][j] 是这个 warp 负责的第 i 行、第 j 列的 16×16
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; ++i)
        #pragma unroll
        for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[FM];
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag[FN];

    for (int t = 0; t < K; t += BK) {
        // 搬运（已写好）：A 小块 128×32 = 4096 个 half = 512 个 uint4，256 个线程每人搬 2 个；B 同理。
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * NTHREADS;               // 0..511
            const int aRow = idx / (BK / 8), aCol = (idx % (BK / 8)) * 8;   // 每行 4 个 uint4
            const int bRow = idx / (BN / 8), bCol = (idx % (BN / 8)) * 8;   // 每行 16 个 uint4
            *reinterpret_cast<uint4*>(&As[aRow][aCol]) = *reinterpret_cast<const uint4*>(&A[aRow * K + t + aCol]);
            *reinterpret_cast<uint4*>(&Bs[bRow][bCol]) = *reinterpret_cast<const uint4*>(&B[(t + bRow) * N + bCol]);
        }
        __syncthreads();

        // ============================== 你要写的部分（二）==============================
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            // TODO 2: 读 FM=4 个 A 片段：第 i 个的左上角是 As 的第 (warpRow0 + i*16) 行、第 kk 列，leading dimension = BK
            // TODO 3: 读 FN=2 个 B 片段：第 j 个的左上角是 Bs 的第 kk 行、第 (warpCol0 + j*16) 列，leading dimension = BN
            // TODO 4: 外积：acc[i][j] += a_frag[i] · b_frag[j]，共 4 × 2 = 8 次 mma_sync
        }
        // ===============================================================================
        __syncthreads();
    }

    // ============================== 你要写的部分（三）==============================
    // TODO 5: 把 8 个累加片段写回 C：acc[i][j] 的左上角是 C 的第 (warpRow0 + i*16) 行、第 (warpCol0 + j*16) 列，
    //         每行 N 个数，wmma::mem_row_major
    // ===============================================================================
}

void launch(const half* A, const half* B, float* C, int M, int N, int K) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "tc02_wmma_smem 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK); exit(1);
    }
    dim3 block(NTHREADS);
    dim3 grid(N / BN, M / BM);
    gemm_wmma_smem<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) { return run_gemm_tc_benchmark("tc02_wmma_smem", launch, argc, argv); }
