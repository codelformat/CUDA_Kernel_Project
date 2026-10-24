// matmul v9：寄存器级双缓冲（片段预取，fragment prefetch）
// 在 v8（warp 级分块）的基础上，只改内层 kk 循环里"读共享内存 → 算外积"的先后顺序。
//
// v8：每个 kk 先从共享内存读 regM、regN（4 条 LDS.128），再用它们做 64 次 FFMA。
//     FFMA 要等 LDS 的数据回来（共享内存延迟约 20–30 个周期），等待期间这个 warp 发不出 FFMA。
// v9：寄存器准备两份：regM[2][TM]、regN[2][TN]，按 kk 的奇偶轮换（和 v5 共享内存双缓冲同一个思路，只是低一层）。
//     算第 kk 步的外积之前，先发出第 kk+1 步的 LDS（写进另一份），
//     这样 LDS 在路上的这段时间，warp 正好在做第 kk 步的 64 次 FFMA，等它们做完数据也到了。
//     每个 BK 块的第 0 步没有"上一步"可以替它预取，所以在 kk 循环之前单独读一次。
//
// 注意：kk 循环是 #pragma unroll 完全展开的，所以 regM[kk % 2] 里的下标在编译时就确定了，仍然放在寄存器里。
//      编译器展开后本来也可能自己把 LDS 往前挪，所以这一课的收益不一定大 —— 先预测，再测。
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

__global__ void __launch_bounds__(NTHREADS) gemm_regprefetch(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ __align__(16) float AsT[2][BK][BM];
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
        fprintf(stderr, "09_reg_prefetch 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK); exit(1);
    }
    dim3 block(NTHREADS);
    dim3 grid(N / BN, M / BM);
    if (EXTRA_SMEM > 0)
        CK(cudaFuncSetAttribute(gemm_regprefetch, cudaFuncAttributeMaxDynamicSharedMemorySize, EXTRA_SMEM));
    gemm_regprefetch<<<grid, block, EXTRA_SMEM>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) { return run_gemm_benchmark("09_reg_prefetch", launch, argc, argv); }
