// CUDA convolution: direct 2D convolution, one thread per output pixel.

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
                              int H, int W, int K)
{
    // row major order x should index the columns due to global memory coalescing within a warp
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    // Filter rows that land inside the image for this output row.
    // Clipping the loop bounds is equivalent to zero padding but keeps
    // the bounds check out of the inner loop.
    if (r < H && c < W)
    {
        const int R = K / 2;
        const int i_lo = max(0, R - r);
        const int i_hi = min(K, H - r + R);
        const int j_lo = max(0, R - c);
        const int j_hi = min(K, W - c + R);

        float acc = 0.0f;
        for (int i = i_lo; i < i_hi; ++i)
        {
            const int y = r + i - R;
            for (int j = j_lo; j < j_hi; ++j)
            {
                acc += k[i * K + j] * in[y * W + (c + j - R)];
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

CudaConv cuda_conv_alloc(int H, int W, int K)
{
    CudaConv c;
    c.H = H;
    c.W = W;
    c.K = K;
    const size_t img = (size_t)H * W * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&c.d_in, img));
    CUDA_CHECK(cudaMalloc((void **)&c.d_out, img));
    CUDA_CHECK(cudaMalloc((void **)&c.d_k, (size_t)K * K * sizeof(float)));
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
    const size_t img = (size_t)c.H * c.W * sizeof(float);
    cudaEvent_t e[4];
    for (int i = 0; i < 4; ++i) e[i] = (cudaEvent_t)c.events[i];

    // set up the execution configuration
    const dim3 block(bx, by);
    const dim3 grid((c.W + block.x - 1) / block.x,
                    (c.H + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventRecord(e[0]));
    CUDA_CHECK(cudaMemcpy(c.d_in, in, img, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.d_k, k, (size_t)c.K * c.K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e[1]));
    conv2d_kernel<<<grid, block>>>(c.d_in, c.d_out, c.d_k, c.H, c.W, c.K);
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
                 int H, int W, int K)
{
    CudaConv c = cuda_conv_alloc(H, W, K);
    cuda_conv_run(c, in, out, k, 32, 8);
    cuda_conv_free(c);
}
