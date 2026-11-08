// Common interface for all 2D convolution implementations.
//
// Operation (see README): cross-correlation, zero padding, same-size output.
//   out[r][c] = sum_{i,j} k[i][j] * in[r + i - R][c + j - R],  R = K / 2
// All arrays are row-major FP32. K must be odd.
#pragma once

void conv2d_seq(const float* in, float* out, const float* k,
                int H, int W, int K);

// OpenMP version. Thread count and schedule come from the usual OpenMP
// controls (OMP_NUM_THREADS / OMP_SCHEDULE or omp_set_*), since the loop
// uses schedule(runtime).
void conv2d_omp(const float* in, float* out, const float* k,
                int H, int W, int K);

// ---------------------------------------------------------------------------
// CUDA version (src/conv2d_cuda.cu). Every function below prints the error
// and exits if any CUDA call fails.

// Same interface as the CPU versions: host pointers in and out. Allocates
// device memory, copies in, runs with a 32x8 thread block, copies back, frees.
void conv2d_cuda(const float* in, float* out, const float* k,
                 int H, int W, int K);

// Lower-level pieces for benchmarking: allocation happens once, outside the
// timed loop, and each stage of a run is timed on the GPU with cudaEvents.
struct CudaConv {
    int H = 0, W = 0, K = 0;
    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_k = nullptr;
    void* events[4] = {};  // cudaEvent_t, opaque so this header stays plain C++
};

struct CudaTimes {
    float h2d_ms = 0;     // input and filter copied host -> device
    float kernel_ms = 0;  // kernel only
    float d2h_ms = 0;     // output copied device -> host
};

void cuda_init();  // create the CUDA context now so later timings exclude it
int cuda_max_threads_per_block();
CudaConv cuda_conv_alloc(int H, int W, int K);
void cuda_conv_free(CudaConv& c);
CudaTimes cuda_conv_run(const CudaConv& c, const float* in, float* out,
                        const float* k, int bx, int by);
