// matmul v8：warp 级分块（block → warp → 线程，三层分块）
// 在 v7（双缓冲 + float4 + A 转置 + 无冲突读取）的基础上，只改"线程负责 C 的哪 64 格"。
//
// v7：16×16 个线程平铺整个 128×128 的 block 小块。一个 warp（ty 两行 × tx 16 列）负责的输出散在
//     2 组 × 8 行 × 128 列里，形状又扁又长（16 × 128）。每个 kk，这个 warp 要从共享内存拿 16 个 A + 128 个 B。
// v8：先把 block 小块切给 8 个 warp，每个 warp 负责一块方正的 64 × 32（warp 小块）；
//     再在 warp 小块里切给 32 个线程，每个线程仍然 8 × 8 = 64 个输出。
//     每个 kk，这个 warp 只要拿 64 个 A + 32 个 B，共 96 个数（v7 是 144 个），做同样的 2048 次乘加。
//     这就是第 1 章外积的道理在 warp 这一层再用一次：区域越"方"，读的数（行 + 列）越少。
//
// 线程排布：block 是一维的 256 个线程。
//   warpId = tid / 32（0..7），lane = tid % 32（0..31）
//   8 个 warp 排成 2 行 × 4 列：warpY = warpId / 4，warpX = warpId % 4
//   warp 小块左上角：warpRow0 = warpY × 64，warpCol0 = warpX × 32
//   warp 内 32 个线程排成 8 行 × 4 列：ly = lane / 4（0..7），lx = lane % 4（0..3）
//   线程负责的行：warpRow0 + ly×4 + 0..3  和  warpRow0 + 32 + ly×4 + 0..3     （8 个 ly × 4 = 32，所以第二组相隔 32）
//   线程负责的列：warpCol0 + lx×4 + 0..3  和  warpCol0 + 16 + lx×4 + 0..3     （4 个 lx × 4 = 16，所以第二组相隔 16）
#include "common.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NTHREADS 256
#define WTM 64                 // warp 小块的行数
#define WTN 32                 // warp 小块的列数
#define WARPS_X (BN / WTN)     // 4
#define LANES_X 4              // warp 内 x 方向的线程数
#define LANES_Y 8              // warp 内 y 方向的线程数
#define WHALF_M (LANES_Y * 4)  // 32：行方向两组之间的间隔
#define WHALF_N (LANES_X * 4)  // 16：列方向两组之间的间隔

// 实验开关：-DEXTRA_SMEM=40960 给每个 block 多占 40 KB 不用的共享内存，
// 使每个 SM 只放得下 1 个 block（16+40=56 KB，两个就超过 100 KB），用来单独测"每 SM 2 个 block"的作用
#ifndef EXTRA_SMEM
#define EXTRA_SMEM 0
#endif

__global__ void __launch_bounds__(NTHREADS) gemm_warptile(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ __align__(16) float AsT[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int tid = threadIdx.x;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // 搬运分工（同 v7）：每个线程每步搬 1 个 A 的 float4、1 个 B 的 float4
    const int aRow = tid / 2,  aCol = (tid % 2) * 4;
    const int bRow = tid / 32, bCol = (tid % 32) * 4;

    // ============================== 你要写的部分（一）==============================
    // TODO 1: 按文件开头的说明，算出 warpId、lane、warpY、warpX、warpRow0、warpCol0、ly、lx
    //         最后得到这个线程两组行的起点 r0、r1，两组列的起点 c0、c1：
    //           r0 = warpRow0 + ly*4，r1 = r0 + WHALF_M；c0 = warpCol0 + lx*4，c1 = c0 + WHALF_N
    int r0 = 0, r1 = 0, c0 = 0, c1 = 0;
    int warpId = tid / 32;
    int lane = tid % 32;
    int warpY = warpId / 4;
    int warpX = warpId % 4;
    int warpRow0 = warpY * 64;
    int warpCol0 = warpX * 32;
    int ly = lane / 4;
    int lx = lane % 4;
    r0 = warpRow0 + ly * 4;
    r1 = r0 + WHALF_M;
    c0 = warpCol0 + lx * 4;
    c1 = c0 + WHALF_N;
    // ===============================================================================

    float acc[TM][TN] = {{0.0f}};
    float regM[TM], regN[TN];
    float4 ldA, ldB;

    {   // 开场：第 0 块（同 v7）
        float4 vA = *reinterpret_cast<const float4*>(&A[aRow * K + aCol]);
        AsT[0][aCol + 0][aRow] = vA.x; AsT[0][aCol + 1][aRow] = vA.y;
        AsT[0][aCol + 2][aRow] = vA.z; AsT[0][aCol + 3][aRow] = vA.w;
        *reinterpret_cast<float4*>(&Bs[0][bRow][bCol]) = *reinterpret_cast<const float4*>(&B[bRow * N + bCol]);
    }
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool has_next = (t + BK < K);
        if (has_next) {
            ldA = *reinterpret_cast<const float4*>(&A[aRow * K + (t + BK) + aCol]);
            ldB = *reinterpret_cast<const float4*>(&B[(t + BK + bRow) * N + bCol]);
        }

        // ============================== 你要写的部分（二）==============================
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // TODO 2: regM[0..3] = AsT[cur][kk][r0 .. r0+3]，regM[4..7] = AsT[cur][kk][r1 .. r1+3]（各一个 float4）
            float4 vA = *reinterpret_cast<const float4*>(&AsT[cur][kk][r0]);
            regM[0] = vA.x; regM[1] = vA.y; regM[2] = vA.z; regM[3] = vA.w;
            vA = *reinterpret_cast<const float4*>(&AsT[cur][kk][r1]);
            regM[4] = vA.x; regM[5] = vA.y; regM[6] = vA.z; regM[7] = vA.w;
            // TODO 3: regN[0..3] = Bs[cur][kk][c0 .. c0+3]， regN[4..7] = Bs[cur][kk][c1 .. c1+3]
            float4 vB = *reinterpret_cast<const float4*>(&Bs[cur][kk][c0]);
            regN[0] = vB.x; regN[1] = vB.y; regN[2] = vB.z; regN[3] = vB.w;
            vB = *reinterpret_cast<const float4*>(&Bs[cur][kk][c1]);
            regN[4] = vB.x; regN[5] = vB.y; regN[6] = vB.z; regN[7] = vB.w;
            // TODO 4: 外积 acc[i][j] += regM[i] * regN[j]
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regM[i] * regN[j];
                }
            }
        }
        // ===============================================================================

        if (has_next) {
            AsT[cur ^ 1][aCol + 0][aRow] = ldA.x; AsT[cur ^ 1][aCol + 1][aRow] = ldA.y;
            AsT[cur ^ 1][aCol + 2][aRow] = ldA.z; AsT[cur ^ 1][aCol + 3][aRow] = ldA.w;
            *reinterpret_cast<float4*>(&Bs[cur ^ 1][bRow][bCol]) = ldB;
        }
        __syncthreads();
        cur ^= 1;
    }

    // ============================== 你要写的部分（三）==============================
    // TODO 5: 写回。acc[i][j] 的行：i < 4 → r0 + i，否则 r1 + (i-4)；列：j < 4 → c0 + j，否则 c1 + (j-4)
    //         用 float4 写：每行 i，acc[i][0..3] 写到列 c0，acc[i][4..7] 写到列 c1
    // ===============================================================================
    for (int i = 0; i < TM; ++i) {
        int row = (i < 4) ? (r0 + i) : (r1 + (i - 4));
        float4 vC;
        vC.x = acc[i][0]; vC.y = acc[i][1]; vC.z = acc[i][2]; vC.w = acc[i][3];
        *reinterpret_cast<float4*>(&C[row * N + c0]) = vC;
        vC.x = acc[i][4]; vC.y = acc[i][5]; vC.z = acc[i][6]; vC.w = acc[i][7];
        *reinterpret_cast<float4*>(&C[row * N + c1]) = vC;
    }
}

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "08_warp_tile 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK); exit(1);
    }
    dim3 block(NTHREADS);
    dim3 grid(N / BN, M / BM);
    if (EXTRA_SMEM > 0)
        CK(cudaFuncSetAttribute(gemm_warptile, cudaFuncAttributeMaxDynamicSharedMemorySize, EXTRA_SMEM));
    gemm_warptile<<<grid, block, EXTRA_SMEM>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) { return run_gemm_benchmark("08_warp_tile", launch, argc, argv); }
