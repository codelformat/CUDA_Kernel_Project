// matmul v10：给 AsT 每行加 padding，消除转置写入的 2 路 bank 冲突（其余同 v9）
//
// 问题：搬运 A 时 aRow = tid/2，aCol = (tid%2)*4。lane 0 和 lane 1 的 aRow 相同、aCol 分别是 0 和 4，
//       写 AsT[aCol + j][aRow] 时地址 = (aCol + j) × 每行长度 + aRow。
//       每行长度 = 128 时，两者相差 4 × 128 = 512，512 % 32 = 0 → 同一个 bank → 每条 STS.32 两轮（2 路冲突）。
// 办法：每行多留 APAD 个不用的位置，每行长度变成 BM + APAD，让上面的差不再是 32 的倍数。
// 代价：共享内存多用 2 × BK × APAD × 4 字节；读取时每行的起点仍要 16 字节对齐，才能继续用 float4 读。
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
// 使每个 SM 只放得下 1 个 block（约 16+40=56 KB，两个就超过 100 KB），用来单独测"每 SM 2 个 block"的作用
// ============================== 你要写的部分 ==============================
// TODO: 选一个 APAD，同时满足 (1) 写 AsT 不冲突 (2) 每行起点 16 字节对齐（float4 读）。在下面写出推导，再填数字。
//   推导：
//
#ifndef APAD
#define APAD 0
#endif
// ===============================================================================

#ifndef EXTRA_SMEM
#define EXTRA_SMEM 0
#endif

__global__ void __launch_bounds__(NTHREADS) gemm_astpad(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ __align__(16) float AsT[2][BK][BM + APAD];   // 每行多 APAD 个空位
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int tid = threadIdx.x;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // 搬运分工（同 v7）：每个线程每步搬 1 个 A 的 float4、1 个 B 的 float4
    const int aRow = tid / 2,  aCol = (tid % 2) * 4;
    const int bRow = tid / 32, bCol = (tid % 32) * 4;

    // 线程 → 输出的对应（同 v8）
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

    float acc[TM][TN] = {{0.0f}};
    float regM[2][TM], regN[2][TN];   // 两份片段：第 kk 步用 regM[kk % 2]
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

        // ============================== 你要写的部分 ==============================
        // 已提供 load_frag(regM[s], regN[s], cur, kk)：把第 kk 步要用的 8 个 A、8 个 B 从共享内存读进第 s 份寄存器
        // （内容和 v8 的 TODO 2、3 完全一样，只是多了"写进哪一份"）。
        #define load_frag(RM, RN, CUR, KK) do {                                                   \
            float4 a0_ = *reinterpret_cast<const float4*>(&AsT[CUR][KK][r0]);                       \
            float4 a1_ = *reinterpret_cast<const float4*>(&AsT[CUR][KK][r1]);                       \
            float4 b0_ = *reinterpret_cast<const float4*>(&Bs[CUR][KK][c0]);                        \
            float4 b1_ = *reinterpret_cast<const float4*>(&Bs[CUR][KK][c1]);                        \
            RM[0] = a0_.x; RM[1] = a0_.y; RM[2] = a0_.z; RM[3] = a0_.w;                             \
            RM[4] = a1_.x; RM[5] = a1_.y; RM[6] = a1_.z; RM[7] = a1_.w;                             \
            RN[0] = b0_.x; RN[1] = b0_.y; RN[2] = b0_.z; RN[3] = b0_.w;                             \
            RN[4] = b1_.x; RN[5] = b1_.y; RN[6] = b1_.z; RN[7] = b1_.w;                             \
        } while (0)

        // TODO 1: 本块第 0 步的片段没人替它预取，在 kk 循环之前读进第 0 份：load_frag(regM[0], regN[0], cur, 0);
        load_frag(regM[0], regN[0], cur, 0);

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // TODO 2: 如果还有下一步（kk + 1 < BK），先把第 kk+1 步的片段读进"另一份"：第 (kk + 1) % 2 份
            if (kk + 1 < BK) {
                load_frag(regM[(kk + 1) % 2], regN[(kk + 1) % 2], cur, kk + 1);
            }

            // TODO 3: 用第 kk % 2 份做外积：acc[i][j] += regM[kk % 2][i] * regN[kk % 2][j]
            for (int i = 0; i < TM; ++i){
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regM[kk % 2][i] * regN[kk % 2][j];
                }
            }

        }
        // 想一想：为什么这一行之后不需要额外的 __syncthreads()？下一块的第 0 步又是在哪里读的？
        // ===============================================================================

        if (has_next) {
            AsT[cur ^ 1][aCol + 0][aRow] = ldA.x; AsT[cur ^ 1][aCol + 1][aRow] = ldA.y;
            AsT[cur ^ 1][aCol + 2][aRow] = ldA.z; AsT[cur ^ 1][aCol + 3][aRow] = ldA.w;
            *reinterpret_cast<float4*>(&Bs[cur ^ 1][bRow][bCol]) = ldB;
        }
        __syncthreads();
        cur ^= 1;
    }

    // 写回（同 v8）
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
        fprintf(stderr, "10_ast_pad 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK); exit(1);
    }
    dim3 block(NTHREADS);
    dim3 grid(N / BN, M / BM);
    if (EXTRA_SMEM > 0)
        CK(cudaFuncSetAttribute(gemm_astpad, cudaFuncAttributeMaxDynamicSharedMemorySize, EXTRA_SMEM));
    gemm_astpad<<<grid, block, EXTRA_SMEM>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) { return run_gemm_benchmark("10_ast_pad", launch, argc, argv); }
