// CUDA convolution: direct 2D convolution, one output pixel at a time.

#include <cuda.h>
#include "conv2d.h"

void FreeDevice(float *dA)
{
    cudaFree(dA);
}

void CopyFromDevice(float *dA, float *hA, size_t size)
{
    cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);
}

void CopyToDevice(const float *hA, float *dA, size_t size)
{
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
}

float *AllocateDevice(size_t size)
{
    float *dA = nullptr;
    cudaMalloc((void **)&dA, size);
    return dA;
}

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

void conv2d_cuda(const float *in, float *out, const float *k,
                 int H, int W, int K)
{
    // allocate and copy input and output buffers to device memory
    float *dI = AllocateDevice(H * W * sizeof(float));
    CopyToDevice(in, dI, H * W * sizeof(float));
    float *dO = AllocateDevice(H * W * sizeof(float));
    // CopyToDevice(out, dO, H * W * sizeof(float));
    float *dK = AllocateDevice(K * K * sizeof(float));
    CopyToDevice(k, dK, K * K * sizeof(float));

    // set up the execution configuration
    dim3 block(32, 8);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    // launch the kernel
    conv2d_kernel<<<grid, block>>>(dI, dO, dK, H, W, K);

    // copy the result back to host memory
    CopyFromDevice(dO, out, H * W * sizeof(float));

    // free the device memory
    FreeDevice(dI);
    FreeDevice(dO);
    FreeDevice(dK);
}
