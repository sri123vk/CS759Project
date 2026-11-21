// CUDA convolution: direct 2D convolution, one thread per output pixel.
//
// Multi-channel input (CHW): each thread sums its K*K window over all C
// input channels, in the same order as conv2d_seq (channel, row, column).

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "conv2d.h"

// Kernel launches don't return errors, and failures inside a kernel only
// surface at the next synchronizing call, so every call is checked.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n  in %s\n",     \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call); \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

__global__ void conv2d_kernel(const float *in, float *out, const float *k,
                              int C, int H, int W, int K)
{
    // row major order x should index the columns due to global memory coalescing within a warp
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    // Filter rows that land inside the image for this output row.
    // Clipping the loop bounds is equivalent to zero padding but keeps
    // the bounds check out of the inner loop. Every channel has the same
    // H x W shape, so the bounds don't depend on the channel.
    if (r < H && c < W)
    {
        const int R = K / 2;
        const int i_lo = max(0, R - r);
        const int i_hi = min(K, H - r + R);
        const int j_lo = max(0, R - c);
        const int j_hi = min(K, W - c + R);
        const size_t plane = (size_t)H * W;  // floats per input channel
        const size_t slice = (size_t)K * K;  // floats per filter channel

        float acc = 0.0f;
        for (int ch = 0; ch < C; ++ch)
        {
            const float *in_ch = in + ch * plane;  // this channel's image
            const float *k_ch = k + ch * slice;    // its filter slice
            for (int i = i_lo; i < i_hi; ++i)
            {
                const int y = r + i - R;
                for (int j = j_lo; j < j_hi; ++j)
                {
                    acc += k_ch[i * K + j] * in_ch[y * W + (c + j - R)];
                }
            }
        }
        out[r * W + c] = acc;
    }
}

void cuda_init()
{
    // The first runtime call creates the context (often 100+ ms).
    CUDA_CHECK(cudaFree(nullptr));
}

int cuda_max_threads_per_block()
{
    int dev = 0, n = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMaxThreadsPerBlock, dev));
    return n;
}

bool cuda_block_supported(int H, int W, int bx, int by)
{
    if (H <= 0 || W <= 0 || bx <= 0 || by <= 0) return false;

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // 64-bit arithmetic so W + bx - 1 cannot overflow for large images.
    const long long threads = (long long)bx * by;
    const long long grid_x = ((long long)W + bx - 1) / bx;
    const long long grid_y = ((long long)H + by - 1) / by;
    return bx <= prop.maxThreadsDim[0] && by <= prop.maxThreadsDim[1] &&
           threads <= prop.maxThreadsPerBlock &&
           grid_x <= prop.maxGridSize[0] && grid_y <= prop.maxGridSize[1];
}

CudaConv cuda_conv_alloc(int C, int H, int W, int K)
{
    CudaConv c;
    c.C = C;
    c.H = H;
    c.W = W;
    c.K = K;
    const size_t img = (size_t)H * W * sizeof(float);  // one channel
    CUDA_CHECK(cudaMalloc((void **)&c.d_in, (size_t)C * img));
    CUDA_CHECK(cudaMalloc((void **)&c.d_out, img));
    CUDA_CHECK(cudaMalloc((void **)&c.d_k, (size_t)C * K * K * sizeof(float)));
    for (void *&e : c.events)
    {
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreate(&ev));
        e = ev;
    }
    return c;
}

void cuda_conv_free(CudaConv &c)
{
    CUDA_CHECK(cudaFree(c.d_in));
    CUDA_CHECK(cudaFree(c.d_out));
    CUDA_CHECK(cudaFree(c.d_k));
    for (void *&e : c.events)
    {
        if (e) CUDA_CHECK(cudaEventDestroy((cudaEvent_t)e));
        e = nullptr;
    }
    c.d_in = c.d_out = c.d_k = nullptr;
}

CudaTimes cuda_conv_run(const CudaConv &c, const float *in, float *out,
                        const float *k, int bx, int by)
{
    const size_t img = (size_t)c.H * c.W * sizeof(float);  // one channel
    cudaEvent_t e[4];
    for (int i = 0; i < 4; ++i) e[i] = (cudaEvent_t)c.events[i];

    // set up the execution configuration
    const dim3 block(bx, by);
    const dim3 grid((c.W + block.x - 1) / block.x,
                    (c.H + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventRecord(e[0]));
    CUDA_CHECK(cudaMemcpy(c.d_in, in, (size_t)c.C * img, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.d_k, k, (size_t)c.C * c.K * c.K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e[1]));
    conv2d_kernel<<<grid, block>>>(c.d_in, c.d_out, c.d_k, c.C, c.H, c.W, c.K);
    CUDA_CHECK(cudaGetLastError());  // bad launch configuration, e.g. > 1024 threads
    CUDA_CHECK(cudaEventRecord(e[2]));
    CUDA_CHECK(cudaMemcpy(out, c.d_out, img, cudaMemcpyDeviceToHost));  // also reports kernel faults
    CUDA_CHECK(cudaEventRecord(e[3]));
    CUDA_CHECK(cudaEventSynchronize(e[3]));

    CudaTimes t;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, e[0], e[1]));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernel_ms, e[1], e[2]));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, e[2], e[3]));
    return t;
}

void conv2d_cuda(const float *in, float *out, const float *k,
                 int C, int H, int W, int K)
{
    CudaConv c = cuda_conv_alloc(C, H, W, K);
    cuda_conv_run(c, in, out, k, 32, 8);
    cuda_conv_free(c);
}
