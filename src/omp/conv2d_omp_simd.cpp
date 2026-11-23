// SIMD openMP: each filter tap is applied across a whole output row at once
// (an axpy into the row), so the innermost loop is long and contiguous and
// vectorizes with omp simd. Each pixel still adds its taps in the same order
// as conv2d_seq (channel, then filter row, then filter column) into one
// running sum: its place in the output row, zeroed once per row.


#include "conv2d.h"
#include <omp.h>
#include <algorithm>

/**
 * Implementation of a 2d convolution using an omp parallel for over rows
 * and omp simd across each output row.
 * Inputs:
 *  in: C input channels, each H x W, channel-major (CHW)
 *  out: output image of size H x W (the channels are summed into it)
 *  k: C filter slices, each K x K, one per input channel
 *  C: number of input channels
 *  H: height of the input image
 *  W: width of the input image
 *  K: size of the kernel (assumed to be odd)
 */
void conv2d_omp_simd(const float* in, float* out, const float* k,
                int C, int H, int W, int K) {
    const int R = K / 2; // radius of the kernel ^^
    const std::size_t plane = (std::size_t)H * W;  // floats per input channel
    const std::size_t slice = (std::size_t)K * K;  // floats per filter channel

    // Parallelization with SIMD vectorization and runtime scheduling
    #pragma omp parallel for schedule(runtime)
    for (int r = 0; r < H; ++r) {   // output rows
        // Zero the output row once, not once per channel: the row holds the
        // running sums for all channels, and zeroing it also clears
        // anything left from a previous call.
        std::fill(out + r * W, out + (r + 1) * W, 0.0f);

        // Loops are reordered so the innermost loop runs along the output row
        // for a single filter tap (axpy form); that's the loop that
        // vectorizes. Filter rows that fall outside the image are clipped.
        const int i_lo = std::max(0, R - r);
        const int i_hi = std::min(K, H - r + R);

        //Channels
        for (int ch = 0; ch < C; ch++) {
            const float* in_ch = in + ch * plane;  // this channel's image
            const float* k_ch = k + ch * slice;    // its filter slice
            // Loop over kernel
            for (int i = i_lo; i < i_hi; ++i) {     //y
                const int y = r + i - R;
                for (int j = 0; j < K; ++j) {       //x
                    const float w = k_ch[i * K + j];   // this channel's weight for tap (i, j)
                    
                    // Calculate range of valid output columns for this kernel position
                    const int c_lo = std::max(0, R - j);
                    const int c_hi = std::min(W, W + R - j);
                    
                    #pragma omp simd    // SIMD vect. time
                    for (int c = c_lo; c < c_hi; ++c) { // output columns
                        out[r * W + c] += w * in_ch[y * W + c + j - R]; // good ol axpy. Slap ts directly into the output.
                    }
                }
            }
        }
    }
}
