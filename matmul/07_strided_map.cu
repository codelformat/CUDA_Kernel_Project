// matmul v7：换一种"线程 → 输出位置"的对应方式，消除读 Bs 的 bank 冲突，并让写回 C 合并。
// v6：线程 (ty, tx) 负责 C 小块里连续的 8 行 × 连续的 8 列：
//       行 ty*8 .. ty*8+7，列 tx*8 .. tx*8+7
//     读 Bs 时相邻线程相隔 8 个数 → tx 和 tx+4 落在同一组 bank（2 路冲突）；
//     写 C 时相邻线程相隔 32 字节 → 一条写指令碰 32 个 sector。
// v7：线程仍然负责 8×8 = 64 个输出，但拆成"相隔 64 的两组、每组 4 个"：
//       行：ty*4 .. ty*4+3  和  64 + ty*4 .. 64 + ty*4+3
//       列：tx*4 .. tx*4+3  和  64 + tx*4 .. 64 + tx*4+3
//     读 Bs 时相邻线程正好相隔 4 个数（一个 float4），8 个线程铺满 32 个 bank，不再冲突；
//     写 C 时 16 个相邻线程写一整段连续的 64 个数。
// 搬运部分和 v6 完全相同（已写好），只改计算时的读取和写回。
#include "common.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NTHREADS ((BM / TM) * (BN / TN))   // 256
#define HALF 64                            // 两组之间的间隔

__global__ void gemm_strided(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ __align__(16) float AsT[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int aRow = tid / 2, aCol = (tid % 2) * 4;
    const int bRow = tid / 32, bCol = (tid % 32) * 4;

    float acc[TM][TN] = {{0.0f}};
    float regM[TM], regN[TN];
    float4 ldA, ldB;

    // ---- 开场：第 0 块（同 v6）----
    {
        float4 vA = *reinterpret_cast<const float4*>(&A[aRow * K + aCol]);
        AsT[0][aCol + 0][aRow] = vA.x; AsT[0][aCol + 1][aRow] = vA.y;
        AsT[0][aCol + 2][aRow] = vA.z; AsT[0][aCol + 3][aRow] = vA.w;
        *reinterpret_cast<float4*>(&Bs[0][bRow][bCol]) = *reinterpret_cast<const float4*>(&B[bRow * N + bCol]);
    }
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool has_next = (t + BK < K);
        if (has_next) {   // 预取下一块（同 v6）
            ldA = *reinterpret_cast<const float4*>(&A[aRow * K + (t + BK) + aCol]);
            ldB = *reinterpret_cast<const float4*>(&B[(t + BK + bRow) * N + bCol]);
        }

        // ============================== 你要写的部分 ==============================
        for (int kk = 0; kk < BK; ++kk) {
            // TODO 1: 读 8 个 A 到 regM：
            //   regM[0..3] = AsT[cur][kk][ty*4 + 0..3]          （一个 float4）
            //   regM[4..7] = AsT[cur][kk][HALF + ty*4 + 0..3]   （另一个 float4）

            // TODO 2: 读 8 个 B 到 regN：
            //   regN[0..3] = Bs[cur][kk][tx*4 + 0..3]
            //   regN[4..7] = Bs[cur][kk][HALF + tx*4 + 0..3]

            // TODO 3: 外积 acc[i][j] += regM[i] * regN[j]（和 v6 一样）
        }
        // ========================================================================

        if (has_next) {   // 写进另一份缓冲（同 v6）
            AsT[cur ^ 1][aCol + 0][aRow] = ldA.x; AsT[cur ^ 1][aCol + 1][aRow] = ldA.y;
            AsT[cur ^ 1][aCol + 2][aRow] = ldA.z; AsT[cur ^ 1][aCol + 3][aRow] = ldA.w;
            *reinterpret_cast<float4*>(&Bs[cur ^ 1][bRow][bCol]) = ldB;
        }
        __syncthreads();
        cur ^= 1;
    }

    // ============================== 你要写的部分 ==============================
    // TODO 4: 写回。acc[i][j] 对应 C 小块的哪一格？
    //   行：i < 4 时是 ty*4 + i；         i >= 4 时是 HALF + ty*4 + (i - 4)
    //   列：j < 4 时是 tx*4 + j；         j >= 4 时是 HALF + tx*4 + (j - 4)
    //   要求用 float4 写：每一行 i，把 acc[i][0..3] 打包成一个 float4 写到列 tx*4 处，
    //                         把 acc[i][4..7] 打包成一个 float4 写到列 HALF + tx*4 处。
    //   （float4 可以这样构造：make_float4(a, b, c, d)）
    // ========================================================================
}

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "07_strided_map 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK);
        exit(1);
    }
    dim3 block(BN / TN, BM / TM);
    dim3 grid(N / BN, M / BM);
    gemm_strided<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    return run_gemm_benchmark("07_strided_map", launch, argc, argv);
}
