// Sequential baseline: direct 2D convolution, one output pixel at a time.

#include "conv2d.h"

#include <algorithm>

void conv2d_seq(const float* in, float* out, const float* k,
                int H, int W, int K) {
    const int R = K / 2;

    for (int r = 0; r < H; ++r) {
        // Filter rows that land inside the image for this output row.
        // Clipping the loop bounds is equivalent to zero padding but keeps
        // the bounds check out of the inner loop.
        const int i_lo = std::max(0, R - r);
        const int i_hi = std::min(K, H - r + R);

        for (int c = 0; c < W; ++c) {
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
    }
}
