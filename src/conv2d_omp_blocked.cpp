// Blocked openMP: each thread works through its rows in blocks of block_size
// output columns. A block's running sums stay together through all K*K
// filter taps, in a small local array the compiler can keep in registers,
// and the loop over the pixels of a block is vectorized with omp simd.
// Each pixel still adds its taps in the same order as conv2d_seq, starting
// from zero, so the output matches it bit for bit (with the default flags).

#include "conv2d.h"
#include <omp.h>
#include <algorithm>

// One output pixel, computed exactly as in conv2d_omp. Used for the columns
// that don't fill a whole block: the left and right edges, where the filter
// has to be clipped, and the remainder after the last full block.
// static: only this file uses it, so the name can't clash with another file's.
static inline void conv_pixel(const float* in, float* out, const float* k,
                              int H, int W, int K, int r, int c) {
    const int R = K / 2;
    const int i_lo = std::max(0, R - r);
    const int i_hi = std::min(K, H - r + R);
    const int j_lo = std::max(0, R - c);
    const int j_hi = std::min(K, W - c + R);

    float acc = 0.0f;
    for (int i = i_lo; i < i_hi; ++i) {
        const int y = r + i - R;
        for (int j = j_lo; j < j_hi; ++j) {
            acc += k[i * K + j] * in[y * W + (c + j - R)];
        }
    }
    out[r * W + c] = acc;
}

/**
 * Implementation of a 2d convolution using an omp parallel for over rows,
 * blocks of output columns within each row, and omp simd within each block.
 * Inputs:
 *  in: input image of size H x W
 *  out: output image of size H x W
 *  k: kernel of size K x K
 *  H: height of the input image
 *  W: width of the input image
 *  K: size of the kernel (assumed to be odd)
 */
void conv2d_omp_blocked(const float* in, float* out, const float* k,
                int H, int W, int K) {
    const int R = K / 2;

    // Pixels per block. constexpr because the array size has to be known at
    // compile time, both to be a real array and for the compiler to have any
    // chance of keeping it in registers. 32 floats = 8 SSE vectors (default
    // flags) or 4 AVX2 vectors (-march=native).
    constexpr int block_size = 32;

    #pragma omp parallel for schedule(runtime)
    for (int r = 0; r < H; ++r) {
        // Filter rows that land inside the image for this output row.
        const int i_lo = std::max(0, R - r);
        const int i_hi = std::min(K, H - r + R);

        // Full blocks, only over the interior columns [R, W - R). There every
        // filter column lands inside the image, so the block needs no
        // clipping and no branch, which is what lets the b loop vectorize.
        int c0 = R;
        for (; c0 + block_size <= W - R; c0 += block_size) {
            float acc[block_size] = {};  // running sums for pixels c0 .. c0 + block_size - 1

            for (int i = i_lo; i < i_hi; ++i) {
                const int y = r + i - R;
                for (int j = 0; j < K; ++j) {
                    const float w = k[i * K + j];                    // one weight per tap
                    const float* src = in + y * W + (c0 + j - R);    // input for pixel c0, this tap

                    #pragma omp simd
                    for (int b = 0; b < block_size; ++b) {
                        acc[b] += w * src[b];
                    }
                }
            }

            #pragma omp simd
            for (int b = 0; b < block_size; ++b) {
                out[r * W + c0 + b] = acc[b];                        // each sum stored once
            }
        }

        // Leftover columns, one pixel at a time: the left edge [0, R), then
        // everything after the last full block (the right edge and any
        // remainder). min/max keep this right when W < 2R + block_size.
        for (int c = 0; c < std::min(R, W); ++c) {
            conv_pixel(in, out, k, H, W, K, r, c);
        }
        for (int c = std::max(c0, R); c < W; ++c) {
            conv_pixel(in, out, k, H, W, K, r, c);
        }
    }
}
