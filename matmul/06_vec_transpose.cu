// matmul v6：在 v5（双缓冲）的基础上，做两件配套的事：
//   ① 显存读改成 float4：一条指令读 4 个连续的 float（第一课 LayerNorm v4 用过）。
//      A 小块 128×8 = 256 个 float4，B 小块 8×128 = 256 个 float4 —— 正好每个线程每步各读 1 个。
//   ② A 小块转置后存进共享内存：AsT[kk][行]。
//      计算时每个 kk 需要的 8 个 A（同一列、8 行）在 AsT 里变成了连续的 8 个数，
//      可以直接用 2 条 128 位读取出来，不再依赖编译器去合并。
#include "common.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NTHREADS ((BM / TM) * (BN / TN))   // 256

// ============================== 你要写的部分 ==============================
__global__ void gemm_vec(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ __align__(16) float AsT[2][BK][BM];   // 转置后的 A：AsT[buf][kk][行]
    __shared__ __align__(16) float Bs[2][BK][BN];    // B 不用转置：Bs[buf][kk][列]

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // 搬运分工（每个线程每步搬 1 个 float4 的 A、1 个 float4 的 B）：
    //   A 小块每行 8 个数 = 2 个 float4：线程负责第 aRow 行、从第 aCol 列开始的 4 个数
    const int aRow = tid / 2, aCol = (tid % 2) * 4;      // aRow: 0..127, aCol: 0 或 4
    //   B 小块每行 128 个数 = 32 个 float4：线程负责第 bRow 行、从第 bCol 列开始的 4 个数
    const int bRow = tid / 32, bCol = (tid % 32) * 4;    // bRow: 0..7,   bCol: 0,4,...,124

    float acc[TM][TN] = {{0.0f}};
    float regM[TM], regN[TN];
    float4 ldA, ldB;                                       // 下一块的数据（预取到寄存器）

    // 用法提示：从地址 p（必须 16 字节对齐）读一个 float4：
    //   float4 v = *reinterpret_cast<const float4*>(p);      v.x v.y v.z v.w 是连续的 4 个数
    //   往共享内存写一个 float4：*reinterpret_cast<float4*>(q) = v;

    // ---- 开场：第 0 块 ----
    // TODO 1: 读 A 的 float4：A 第 aRow 行、第 aCol..aCol+3 列（第 0 块，t = 0）
    //         转置写入：AsT[0][aCol+0][aRow] = v.x; AsT[0][aCol+1][aRow] = v.y; ... 共 4 个
    float4 vA = *reinterpret_cast<const float4*>(&A[aRow * K + aCol]);
    AsT[0][aCol + 0][aRow] = vA.x;
    AsT[0][aCol + 1][aRow] = vA.y;
    AsT[0][aCol + 2][aRow] = vA.z;
    AsT[0][aCol + 3][aRow] = vA.w;

    // TODO 2: 读 B 的 float4：B 第 bRow 行、第 bCol..bCol+3 列；整个 float4 写入 &Bs[0][bRow][bCol]
    float4 vB = *reinterpret_cast<const float4*>(&B[bRow * N + bCol]);
    Bs[0][bRow][bCol + 0] = vB.x;
    Bs[0][bRow][bCol + 1] = vB.y;
    Bs[0][bRow][bCol + 2] = vB.z;
    Bs[0][bRow][bCol + 3] = vB.w;

    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool has_next = (t + BK < K);

        // TODO 3: 如果 has_next：预取下一块到 ldA、ldB（A 的列偏移是 t + BK + aCol；B 的行是 t + BK + bRow）
        //         【检查：两处都加了 t + BK 吗？】
        if (has_next) {
            ldA = *reinterpret_cast<const float4*>(&A[aRow * K + (t + BK) + aCol]);
            ldB = *reinterpret_cast<const float4*>(&B[(t + BK + bRow) * N + bCol]);
        }

        // TODO 4: 计算。对 kk = 0..BK-1：
        //   regM[0..3] = AsT[cur][kk][ty*TM + 0..3] 用一个 float4 读；regM[4..7] 用另一个 float4 读（起点 +4）
        //   regN[0..3] = Bs[cur][kk][tx*TN + 0..3] 用一个 float4 读；regN[4..7] 同理
        //   然后外积：acc[i][j] += regM[i] * regN[j]
        for (int kk = 0; kk < BK; ++kk) {
            float4 vM0 = *reinterpret_cast<float4*>(&AsT[cur][kk][ty * TM + 0]);
            float4 vM1 = *reinterpret_cast<float4*>(&AsT[cur][kk][ty * TM + 4]);
            float4 vN0 = *reinterpret_cast<float4*>(&Bs[cur][kk][tx * TN + 0]);
            float4 vN1 = *reinterpret_cast<float4*>(&Bs[cur][kk][tx * TN + 4]);

            regM[0] = vM0.x; regM[1] = vM0.y; regM[2] = vM0.z; regM[3] = vM0.w;
            regM[4] = vM1.x; regM[5] = vM1.y; regM[6] = vM1.z; regM[7] = vM1.w;

            regN[0] = vN0.x; regN[1] = vN0.y; regN[2] = vN0.z; regN[3] = vN0.w;
            regN[4] = vN1.x; regN[5] = vN1.y; regN[6] = vN1.z; regN[7] = vN1.w;

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regM[i] * regN[j];
                }
            }
        }

        // TODO 5: 如果 has_next：把 ldA 转置写进 AsT[cur ^ 1]，把 ldB 整个写进 Bs[cur ^ 1]
        if (has_next) {
            AsT[cur ^ 1][aCol + 0][aRow] = ldA.x;
            AsT[cur ^ 1][aCol + 1][aRow] = ldA.y;
            AsT[cur ^ 1][aCol + 2][aRow] = ldA.z;
            AsT[cur ^ 1][aCol + 3][aRow] = ldA.w;

            Bs[cur ^ 1][bRow][bCol + 0] = ldB.x;
            Bs[cur ^ 1][bRow][bCol + 1] = ldB.y;
            Bs[cur ^ 1][bRow][bCol + 2] = ldB.z;
            Bs[cur ^ 1][bRow][bCol + 3] = ldB.w;
        }

        // TODO 6: 同步一次，cur ^= 1
        __syncthreads();
        cur ^= 1;
    }

    // TODO 7: 写回 64 个结果（同 v4）。可选加分：每行 8 个数用 2 个 float4 写回。
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            C[(ty * TM + i) * N + (tx * TN + j)] = acc[i][j];
        }
    }
}
// ========================================================================

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "06_vec_transpose 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, BK);
        exit(1);
    }
    dim3 block(BN / TN, BM / TM);
    dim3 grid(N / BN, M / BM);
    gemm_vec<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    return run_gemm_benchmark("06_vec_transpose", launch, argc, argv);
}
