// matmul 阶段 B · tc01：第一次用张量核心（WMMA API）（阶段 A 补课完成后再做）
// 张量核心是"二维"的乘加单元：一个 warp 一起发出一条 mma，硬件一次算完一个 16×16×16 的小矩阵乘：
//     c(16×16) += a(16×16) · b(16×16)          （a、b 是 fp16，c 是 fp32）
// 对比：一条 FFMA 指令只做 32 次乘加（每线程 1 次）；一次 16×16×16 是 4096 次乘加。
//
// WMMA 用"片段（fragment）"表示这三个小矩阵。片段的数据分散在 warp 里 32 个线程的寄存器中，
// 具体谁拿哪几个数由硬件规定、对程序员不透明 —— 所以 load / mma / store 都必须整个 warp 一起调用（_sync）。
//
// 这一版是"最朴素的张量核心 GEMM"，对应阶段 A 的 v1：
//   每个 warp 负责 C 的一个 16×16 小块，沿 K 每次走 16，直接从显存把 a、b 读进片段，不用共享内存。
//   一个 block 有 4 个 warp，排成 2×2，负责 C 的 32×32。
#include "common_tc.cuh"
#include <mma.h>
using namespace nvcuda;

#define WM 16
#define WN 16
#define WK 16
#define WARPS_X 2        // block 里 x 方向几个 warp
#define WARPS_Y 2        // block 里 y 方向几个 warp

// ============================== 你要写的部分 ==============================
// 要求 M、N 是 16 的倍数（launch 里检查），K 是 16 的倍数。
__global__ void gemm_wmma_naive(const half* A, const half* B, float* C, int M, int N, int K) {
    const int warpId = threadIdx.x / 32;          // 0..3：block 里第几个 warp
    const int warpX  = warpId % WARPS_X;          // 这个 warp 在 block 里的列号（0..1）
    const int warpY  = warpId / WARPS_X;          // 这个 warp 在 block 里的行号（0..1）

    // TODO 1: 这个 warp 负责 C 的哪个 16×16 小块？算出它左上角的行 row0、列 col0。
    //         提示：block 负责 (WARPS_Y*16) × (WARPS_X*16) 的区域，用"第几份 × 每份大小"。
    int row0 = 0, col0 = 0;

    // 三个片段：a 是行主序的 16×16 half，b 是行主序的 16×16 half，c 是 16×16 float 累加器
    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;

    // TODO 2: 把 c_frag 清零（wmma::fill_fragment）

    for (int k = 0; k < K; k += WK) {
        // TODO 3: 从显存读 a 的 16×16 小块：左上角是 A 的第 row0 行、第 k 列；A 每行有 K 个数（leading dimension = K）
        //         wmma::load_matrix_sync(a_frag, 指向左上角的指针, leading dimension);
        // TODO 4: 读 b 的 16×16 小块：左上角是 B 的第 k 行、第 col0 列；B 每行有 N 个数
        // TODO 5: c_frag += a_frag · b_frag   （wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)）
    }

    // TODO 6: 把 c_frag 写回 C：左上角 C 的第 row0 行、第 col0 列，每行 N 个数，行主序（wmma::mem_row_major）
}
// ========================================================================

void launch(const half* A, const half* B, float* C, int M, int N, int K) {
    if (M % (WARPS_Y * WM) || N % (WARPS_X * WN) || K % WK) {
        fprintf(stderr, "tc01_wmma 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", WARPS_Y * WM, WARPS_X * WN, WK); exit(1);
    }
    dim3 block(32 * WARPS_X * WARPS_Y);                       // 128 个线程 = 4 个 warp
    dim3 grid(N / (WARPS_X * WN), M / (WARPS_Y * WM));
    gemm_wmma_naive<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) { return run_gemm_tc_benchmark("tc01_wmma_naive", launch, argc, argv); }
