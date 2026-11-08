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

//optimized version for the large size matrices
//image rectangle shape handled
//zero-padding,kernel square matrix with odd size
void convolve(
    const float *image,
    float *output,
    std::size_t height,
    std::size_t width,
    const float *mask,
    std::size_t m
);