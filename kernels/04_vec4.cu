// 第 11 课：v4 —— float4 向量化的 load / store。
// 相对 v3 只改一件事：访存指令的宽度，从每个 thread 一次 4 B 变成一次 16 B（float4）。
// 其余不变：一个 block 一行、寄存器缓存、共享内存归约、三步（均值 -> 方差 -> 输出）。
#include "common.cuh"
#include "block_reduce.cuh"   // block_reduce_sum

// VEC = 每个 thread 负责的 float4 个数 = H / (4 * blockDim.x)，仍然是编译期常量，v[VEC] 常驻寄存器。
template <int VEC>
__global__ void layernorm_vec4(const float* x, const float* gamma, const float* beta,
                               float* y, int N, int H, float eps) {
    __shared__ float smem[MAX_THREADS];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    // 把 float* 当成 float4* 来看：下标的单位从"一个 float（4 B）"变成"一个 float4（16 B）"，一行有 H/4 个 float4。
    // float4 的地址必须 16 B 对齐：cudaMalloc 的起始地址满足，每行的起点 row*H*4 B 在 H % 4 == 0 时也满足。
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * H);
    float4* yr = reinterpret_cast<float4*>(y + (size_t)row * H);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    (void)N;

    float4 v[VEC];   // 这个 thread 的 VEC 个 float4，即 4*VEC 个 float，常驻寄存器

    // ============================== 你要写的部分 ==============================
    // 约定：第 i 个 float4 的下标 k = tid + i * blockDim.x（单位是 float4）。
    //       它对应的列号是 4*k, 4*k+1, 4*k+2, 4*k+3；gamma、beta 同样用 g4[k]、b4[k] 一次取 4 个。
    //       float4 的四个分量叫 .x .y .z .w。

    // TODO 1: 第一遍。对 i = 0 .. VEC-1：v[i] = xr[k]（一条指令读 16 B），把 4 个分量都累加进部分和；
    //         block_reduce_sum -> 除以 H（注意是 H，不是 H/4）得到 mean。
    float mean = 0.f;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        v[i] = xr[tid + i * blockDim.x];
        mean += v[i].x + v[i].y + v[i].z + v[i].w;
    }
    mean = block_reduce_sum(mean, smem) / H;

    // TODO 2: 第二遍。只用 v[i]：4 个分量各算 (分量 - mean)^2，累加 -> 归约 -> 除以 H 得到 var。
    float var = 0.f;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        float diff = v[i].x - mean;
        var += diff * diff;
        diff = v[i].y - mean;
        var += diff * diff;
        diff = v[i].z - mean;
        var += diff * diff;
        diff = v[i].w - mean;
        var += diff * diff;
    }
    var = block_reduce_sum(var, smem) / H;

    // TODO 3: 第三遍。对每个 i：取 g = g4[k]、b = b4[k]，4 个分量各自算
    //         (分量 - mean) / sqrtf(var + eps) * g.分量 + b.分量，拼成一个 float4，一次写回 yr[k]。
    // ========================================================================
    (void)v; (void)mean; (void)var; (void)yr; (void)g4; (void)b4; (void)xr; (void)eps; (void)tid;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int k = tid + i * blockDim.x;
        float4 g = g4[k];
        float4 b = b4[k];
        float4 out;
        out.x = (v[i].x - mean) / sqrtf(var + eps) * g.x + b.x;
        out.y = (v[i].y - mean) / sqrtf(var + eps) * g.y + b.y;
        out.z = (v[i].z - mean) / sqrtf(var + eps) * g.z + b.z;
        out.w = (v[i].w - mean) / sqrtf(var + eps) * g.w + b.w;
        yr[k] = out;
    }
}

void launch(const float* x, const float* gamma, const float* beta, float* y,
            int N, int H, float eps, int threads) {
    if (threads < 32 || threads > MAX_THREADS || (threads & (threads - 1)) != 0 || H % (4 * threads) != 0) {
        fprintf(stderr, "threads 必须是 2 的幂、在 [32, %d] 之间，且 4*threads 能整除 H=%d，现在是 %d\n",
                MAX_THREADS, H, threads);
        exit(1);
    }
    switch (H / (4 * threads)) {
        case 1: layernorm_vec4<1><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        case 2: layernorm_vec4<2><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        case 4: layernorm_vec4<4><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        case 8: layernorm_vec4<8><<<N, threads>>>(x, gamma, beta, y, N, H, eps); break;
        default:
            fprintf(stderr, "H / (4*threads) = %d，不在支持的 {1, 2, 4, 8} 里\n", H / (4 * threads));
            exit(1);
    }
    CK(cudaGetLastError());
}

int main(int argc, char** argv) {
    return run_benchmark("04_vec4", launch, argc, argv);
}
