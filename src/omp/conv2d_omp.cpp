// Naive openMP parallelized: 2D convolution using openMP.
// This is a direct parallelization of the sequential implementation, with the outer loop parallelized.


#include "conv2d.h"
#include <omp.h>
#include <algorithm>

/**
 * Implementation of a 2d convolution using a single 
 * omp parallel for directive.
 * Inputs:
 *  in: C input channels, each H x W, channel-major (CHW)
 *  out: output image of size H x W (the channels are summed into it)
 *  k: C filter slices, each K x K, one per input channel
 *  C: number of input channels
 *  H: height of the input image
 *  W: width of the input image
 *  K: size of the kernel (assumed to be odd)
 */
void conv2d_omp(const float* in, float* out, const float* k,
                int C, int H, int W, int K) {
    const int R = K / 2;
    const std::size_t plane = (std::size_t)H * W;  // floats per input channel
    const std::size_t slice = (std::size_t)K * K;  // floats per filter channel
    
    // A "crazy" parallelization: row-wise parallelization
    #pragma omp parallel for schedule(runtime)
    for (int r = 0; r < H; ++r) {
        // Filter rows that land inside the image for this output row.
        // Clipping the loop bounds is equivalent to zero padding but keeps
        // the bounds check out of the inner loop.
        const int i_lo = std::max(0, R - r);
        const int i_hi = std::min(K, H - r + R);

        for (int c = 0; c < W; ++c) {
            const int j_lo = std::max(0, R - c);
            const int j_hi = std::min(K, W - c + R);

            float acc = 0.0f;  // one running sum across all channels, as in conv2d_seq
            for (int ch = 0; ch < C; ++ch) {
                const float* in_ch = in + ch * plane;  // this channel's image
                const float* k_ch = k + ch * slice;    // its filter slice
                for (int i = i_lo; i < i_hi; ++i) {
                    const int y = r + i - R;
                    for (int j = j_lo; j < j_hi; ++j) {
                        acc += k_ch[i * K + j] * in_ch[y * W + (c + j - R)];
                    }
                }
            }
            out[r * W + c] = acc;
        }
    }
}
