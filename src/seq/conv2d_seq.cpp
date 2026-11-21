// Sequential baseline: direct 2D convolution, one output pixel at a time.
//
// Multi-channel input (CHW): each output pixel sums its K*K window over all
// C input channels, using that channel's slice of the filter.

#include "conv2d.h"

#include <algorithm>
#include <cstddef>

void conv2d_seq(const float* in, float* out, const float* k,
                int C, int H, int W, int K) {
    const int R = K / 2;
    const std::size_t plane = (std::size_t)H * W;  // floats per input channel
    const std::size_t slice = (std::size_t)K * K;  // floats per filter channel

    for (int r = 0; r < H; ++r) {
        // Filter rows that land inside the image for this output row.
        // Clipping the loop bounds is equivalent to zero padding but keeps
        // the bounds check out of the inner loop. Every channel has the same
        // H x W shape, so the bounds don't depend on the channel.
        const int i_lo = std::max(0, R - r);
        const int i_hi = std::min(K, H - r + R);

        for (int c = 0; c < W; ++c) {
            const int j_lo = std::max(0, R - c);
            const int j_hi = std::min(K, W - c + R);

            float acc = 0.0f;
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
