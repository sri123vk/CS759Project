// simd OMP version of conv2d.cpp:


#include "conv2d.h"
#include <omp.h>
#include <algorithm>

/**
 * Implementation of a 2d convolution using a single 
 * omp parallel for directive.
 * Inputs:
 *  in: input image of size H x W
 *  out: output image of size H x W
 *  k: kernel of size K x K
 *  H: height of the input image
 *  W: width of the input image
 *  K: size of the kernel (assumed to be odd)
 */
void conv2d_omp_simd(const float* in, float* out, const float* k,
                int H, int W, int K) {
    const int R = K / 2; // radius of the kernel ^^

    // Parallelization with SIMD vectorization and runtime scheduling
    #pragma omp parallel for schedule(runtime)
    for (int r = 0; r < H; ++r) {   // output rows
        //Zero the output row to avoid accumulation of previous values
        std::fill(out + r * W, out + (r + 1) * W, 0.0f); // i

        // SIMD vectorization via reordering
        // loops and performing kernel-wise 
        // vectorization.
        const int i_lo = std::max(0, R - r);
        const int i_hi = std::min(K, H - r + R);

        // Loop over kernel
        for (int i = i_lo; i < i_hi; ++i) {     //y
            const int y = r + i - R;
            for (int j = 0; j < K; ++j) {       //x
                const float w = k[i * K + j];   // indexing into kernel weight
                
                // Calculate range of valid output columns for this kernel position
                const int c_lo = std::max(0, R - j);
                const int c_hi = std::min(W, W + R - j);
                
                #pragma omp simd    // SIMD vect. time
                for (int c = c_lo; c < c_hi; ++c) { // output columns
                    out[r * W + c] += w * in[y * W + c + j - R]; // good ol axpy. Slap ts directly into the output.
                }
            }
        }
    }
}
