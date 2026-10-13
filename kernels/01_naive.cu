// 第 2 课：最朴素的 LayerNorm kernel —— 一个 thread 负责一整行。
// 目标只有一个：结果正确。性能现在完全不用管。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

// ============================== 你要写的部分 ==============================
// 数据是 row-major：第 r 行第 j 个元素在 x[r * H + j]。
__global__ void layernorm_naive(const float* x, const float* gamma, const float* beta,
                                float* y, int N, int H, float eps) {
    // TODO 1: 算出这个 thread 负责哪一行（用 blockIdx.x / blockDim.x / threadIdx.x）
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO 2: 越界保护：grid 是向上取整算出来的，N 不一定是 blockDim 的整数倍
    if (row >= N) return;
    // TODO 3: 第一遍扫描这一行，求均值
    int offset = row * H;
    float mean = 0;
    for (int j = 0; j < H; ++j){
        mean += x[offset + j];
    }
    mean /= H;
    // TODO 4: 第二遍扫描这一行，求方差（除以 H）
    float var = 0;
    for (int j = 0; j < H; ++j){
        float diff = x[offset + j] - mean;
        var += diff * diff;
    }
    var /= H;
    // TODO 5: 第三遍扫描这一行，算出 y 写回去（别忘了 gamma、beta、eps）
    for (int j = 0; j < H; ++j){
        y[offset + j] = (x[offset + j] - mean) / sqrt(var + eps) * gamma[j] + beta[j];
    }
}
// ========================================================================

// ---------------- 下面是测试框架（host 代码），先不用逐行看懂 ----------------
// 流程：读数据 -> 拷到 GPU -> 启动 kernel -> 拷回 CPU -> 和 PyTorch 的结果对比

static std::string path_of(const char* dir, const char* name) {
    return std::string(dir) + "/" + name;
}

static std::vector<float> read_bin(const std::string& path, size_t count) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "打不开 %s\n  请先在本机运行 python ref/gen_data.py，再运行 scripts/push_data.sh\n", path.c_str());
        exit(1);
    }
    std::vector<float> v(count);
    size_t got = fread(v.data(), sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "%s: 期望 %zu 个 float，实际读到 %zu 个\n", path.c_str(), count, got);
        exit(1);
    }
    return v;
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "data";

    int N = 0, H = 0;
    float eps = 0.f;
    {
        FILE* f = fopen(path_of(dir, "meta.txt").c_str(), "r");
        if (!f || fscanf(f, "%d %d %f", &N, &H, &eps) != 3) {
            fprintf(stderr, "读不到 %s/meta.txt\n", dir);
            return 1;
        }
        fclose(f);
    }
    const size_t nx = (size_t)N * H;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)   N=%d H=%d eps=%g\n", prop.name, prop.major, prop.minor, N, H, eps);

    std::vector<float> hx = read_bin(path_of(dir, "x.bin"), nx);
    std::vector<float> hgamma = read_bin(path_of(dir, "gamma.bin"), H);
    std::vector<float> hbeta = read_bin(path_of(dir, "beta.bin"), H);
    std::vector<float> href = read_bin(path_of(dir, "y_ref.bin"), nx);
    std::vector<float> hy(nx, 0.f);

    float *d_x, *d_gamma, *d_beta, *d_y;
    CK(cudaMalloc(&d_x, nx * sizeof(float)));
    CK(cudaMalloc(&d_gamma, H * sizeof(float)));
    CK(cudaMalloc(&d_beta, H * sizeof(float)));
    CK(cudaMalloc(&d_y, nx * sizeof(float)));
    CK(cudaMemcpy(d_x, hx.data(), nx * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_gamma, hgamma.data(), H * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_beta, hbeta.data(), H * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_y, 0, nx * sizeof(float)));  // 清零：kernel 没写到的位置会因此露馅

    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;
    printf("launch: %d blocks x %d threads = %d threads, for %d rows\n", blocks, threads, blocks * threads, N);
    layernorm_naive<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, N, H, eps);
    CK(cudaGetLastError());       // 检查启动参数是否合法
    CK(cudaDeviceSynchronize());  // 等 kernel 真正跑完，并捕获运行期错误

    CK(cudaMemcpy(hy.data(), d_y, nx * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    size_t worst = 0, n_nan = 0;
    for (size_t i = 0; i < nx; ++i) {
        if (std::isnan(hy[i])) { ++n_nan; continue; }
        float d = fabsf((float)hy[i] - (float)href[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
    }
    const float tol = 1e-5f;
    const bool pass = (n_nan == 0) && (max_abs < tol);
    printf("max_abs_err = %.3e (row %zu, col %zu)   nan = %zu   -> %s (tol %.0e)\n",
           max_abs, worst / H, worst % H, n_nan, pass ? "PASS" : "FAIL", tol);

    // TODO 8: 热身。直接再启动 10 次同一个 kernel（参数和上面一样），不计时。
    //         循环结束后调一次 cudaDeviceSynchronize()，确保热身真的做完了。
    for (int i = 0; i < 10; i ++) {
        layernorm_naive<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, N, H, eps);
    }
    CK(cudaDeviceSynchronize());
    // TODO 9: 正式测量 100 次，每次独立计时，结果存进 std::vector<float> times(100)：
    //         每一轮：cudaEventRecord(start) -> 启动 kernel -> cudaEventRecord(stop)
    //                 -> cudaEventSynchronize(stop) -> cudaEventElapsedTime(&times[i], start, stop)
    //         注意：只围住 kernel 启动这一行，不要把 cudaMemcpy / cudaMemset 放进去。
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));
    std::vector<float> times(100);
    for (int i = 0; i < 100; i ++) {
        CK(cudaEventRecord(start));
        layernorm_naive<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, N, H, eps);
        CK(cudaEventRecord(stop));
        CK(cudaEventSynchronize(stop));
        float elapsed = 0.f;
        CK(cudaEventElapsedTime(&elapsed, start, stop));
        times[i] = elapsed;
    }
    // TODO 10: std::sort(times.begin(), times.end())（需要在文件开头 #include <algorithm>），
    //          打印中位数 times[50] 和最小值 times[0]，单位 ms。
    std::sort(times.begin(), times.end());
    printf("median time = %.3f ms, min time = %.3f ms\n", times[50], times[0]);
    // TODO 11: 释放事件：cudaEventDestroy(start) / (stop)。
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    // 先别算带宽，下一课再做。
    // ===============================================================================
    

    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_y);
    return pass ? 0 : 1;
}
