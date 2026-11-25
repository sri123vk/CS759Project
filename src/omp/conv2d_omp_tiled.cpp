// Tiled OpenMP (review: "block caching" / CPU locality).
//
// conv2d_omp_blocked computes one output row at a time. Here a thread
// computes a tile of TILE_ROWS output rows x TILE_COLS columns at once:
//  - the K + TILE_ROWS - 1 input rows a tile reads (per channel) are reused,
//    while in L1, by all the tile's output rows that need them, instead of
//    being brought back once per output row;
//  - the tile's TILE_ROWS * TILE_COLS running sums are independent of each
//    other, so the CPU can overlap their additions instead of waiting on one
//    chain of dependent adds per vector register.
//
// With C input channels the tile's sums are either carried across all
// channels in registers, or, for large planes, a strip is swept once per
// channel with the sums kept in the output in between (see conv2d_omp_tiled
// for when). Either way each pixel adds its taps in the same order as
// conv2d_seq (channel, then filter row, then filter column, starting from
// zero), so the output matches it bit for bit when the compiler does not
// contract the multiply-adds into FMAs.

#include "conv2d.h"
#include <omp.h>
#include <algorithm>
#include <cstddef>

namespace {

// Output rows and columns per tile. 4 x 16 = 64 running sums: 8 AVX2
// registers or 16 SSE registers.
constexpr int TILE_ROWS = 4;
constexpr int TILE_COLS = 16;

// One output pixel, computed as in conv2d_seq. Used for the columns that
// don't fill a whole tile: the left and right edges, where the filter has to
// be clipped, and the remainder after the last full tile.
inline void conv_pixel(const float* in, float* out, const float* k,
                       int C, int H, int W, int K, int r, int c) {
    const int R = K / 2;
    const std::size_t plane = (std::size_t)H * W;  // floats per input channel
    const std::size_t slice = (std::size_t)K * K;  // floats per filter channel

    const int i_lo = std::max(0, R - r);
    const int i_hi = std::min(K, H - r + R);
    const int j_lo = std::max(0, R - c);
    const int j_hi = std::min(K, W - c + R);

    float acc = 0.0f;  // one running sum across all channels
    for (int ch = 0; ch < C; ++ch) {
        const float* in_ch = in + ch * plane;
        const float* k_ch = k + ch * slice;
        for (int i = i_lo; i < i_hi; ++i) {
            const int y = r + i - R;
            for (int j = j_lo; j < j_hi; ++j) {
                acc += k_ch[i * K + j] * in_ch[y * W + (c + j - R)];
            }
        }
    }
    out[r * W + c] = acc;
}

// One channel of a tile whose rows are all at least R away from the top and
// bottom, so every filter row applies. Constant trip counts and no
// branches, so the compiler can keep acc in registers.
inline void tile_interior(const float* in_ch, const float* k_ch, int W, int K,
                          int r0, int c0, float (&acc)[TILE_ROWS][TILE_COLS]) {
    const int R = K / 2;
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            const float w = k_ch[i * K + j];
            // Input for output row r0; row r0 + q is q * W further on.
            const float* src = in_ch + (r0 + i - R) * W + (c0 + j - R);
            #pragma GCC unroll 8
            for (int q = 0; q < TILE_ROWS; ++q) {
                #pragma omp simd
                for (int b = 0; b < TILE_COLS; ++b) {
                    acc[q][b] += w * src[q * W + b];
                }
            }
        }
    }
}

// One channel of a tile in the top or bottom strips, where some filter rows
// fall outside the image (and the last strip may have fewer than TILE_ROWS
// rows). Walks the input rows y the tile touches; output row r0 + q uses
// row y with filter row i = y - (r0 + q) + R when 0 <= i < K. For each pixel,
// i still increases with y, so within the channel the taps are added in
// seq's order. The caller loops over channels outside this function, which
// keeps the channel-major order across channels.
inline void tile_edge(const float* in_ch, const float* k_ch, int H, int W, int K,
                      int r0, int rows, int c0,
                      float (&acc)[TILE_ROWS][TILE_COLS]) {
    const int R = K / 2;
    const int y_lo = std::max(0, r0 - R);
    const int y_hi = std::min(H, r0 + rows + R);
    for (int y = y_lo; y < y_hi; ++y) {
        const int q_lo = std::max(0, y - r0 + R - (K - 1));
        const int q_hi = std::min(rows, y - r0 + R + 1);
        const float* src = in_ch + y * W + (c0 - R);
        for (int j = 0; j < K; ++j) {
            for (int q = q_lo; q < q_hi; ++q) {
                const float w = k_ch[(y - r0 - q + R) * K + j];
                #pragma omp simd
                for (int b = 0; b < TILE_COLS; ++b) {
                    acc[q][b] += w * src[j + b];
                }
            }
        }
    }
}

}  // namespace

/**
 * 2d convolution with an omp parallel for over strips of TILE_ROWS output
 * rows; each strip is computed in tiles of TILE_ROWS x TILE_COLS pixels.
 * Inputs:
 *  in: C input channels, each H x W, channel-major (CHW)
 *  out: output image of size H x W (the channels are summed into it)
 *  k: C filter slices, each K x K, one per input channel
 *  C: number of input channels
 *  H: height of the input image
 *  W: width of the input image
 *  K: size of the kernel (assumed to be odd)
 */
void conv2d_omp_tiled(const float* in, float* out, const float* k,
                      int C, int H, int W, int K) {
    const int R = K / 2;
    const std::size_t plane = (std::size_t)H * W;
    const std::size_t slice = (std::size_t)K * K;
    const int strips = (H + TILE_ROWS - 1) / TILE_ROWS;

    // Channels per pass over a strip (see the loop below).
    //  chunk = C: every channel for one tile, then the next tile. The sums
    //    stay in registers for the whole tile; best when the C planes are
    //    small enough to stay in cache together.
    //  chunk = 1: one channel across the whole strip, then the next. A
    //    channel's tiles walk along the same K + 3 input rows in order, but
    //    the sums go through `out` between channels. Best once each plane is
    //    large: with chunk = C every 16 columns jump between C planes that
    //    are H*W floats apart, which thrashes the cache.
    // Measured on Euler (16 threads, see the omp_tiled loop-order note in
    // BENCHMARK.md): chunk = 1 is up to 1.55x faster at 1024x1024 with
    // K >= 5 and C >= 16, but slower for K = 3 and for images of 512x512 or
    // less, where the extra trips through `out` cost more than they save.
    const bool by_channel = C > 1 && K >= 5 && plane >= (std::size_t(1) << 19);
    const int chunk = by_channel ? 1 : C;

    #pragma omp parallel for schedule(runtime)
    for (int s = 0; s < strips; ++s) {
        const int r0 = s * TILE_ROWS;                  // first output row of the strip
        const int rows = std::min(TILE_ROWS, H - r0);  // fewer in the last strip
        const bool interior = rows == TILE_ROWS && r0 >= R && r0 + TILE_ROWS + R <= H;

        // Full tiles, only over the interior columns [R, W - R), where every
        // filter column lands inside the image (as in conv2d_omp_blocked).
        // c_end is where they stop; the columns after it are leftovers.
        const int full = W - 2 * R >= TILE_COLS ? (W - 2 * R) / TILE_COLS : 0;
        const int c_end = R + full * TILE_COLS;

        // The channels are taken `chunk` at a time: each pass sweeps the
        // strip's tiles for those channels, and the partial sums wait in
        // `out` between passes. Reading them back continues the same chain of
        // float additions, so each pixel is still summed in seq's order
        // (channel, filter row, filter column) whatever the chunk.
        for (int ch0 = 0; ch0 < C; ch0 += chunk) {
            const int ch1 = std::min(C, ch0 + chunk);
            for (int c0 = R; c0 < c_end; c0 += TILE_COLS) {
                float acc[TILE_ROWS][TILE_COLS] = {};
                if (ch0 > 0) {
                    for (int q = 0; q < rows; ++q) {
                        #pragma omp simd
                        for (int b = 0; b < TILE_COLS; ++b) {
                            acc[q][b] = out[(r0 + q) * W + c0 + b];
                        }
                    }
                }
                for (int ch = ch0; ch < ch1; ++ch) {
                    const float* in_ch = in + ch * plane;
                    const float* k_ch = k + ch * slice;
                    if (interior) tile_interior(in_ch, k_ch, W, K, r0, c0, acc);
                    else          tile_edge(in_ch, k_ch, H, W, K, r0, rows, c0, acc);
                }
                for (int q = 0; q < rows; ++q) {
                    #pragma omp simd
                    for (int b = 0; b < TILE_COLS; ++b) {
                        out[(r0 + q) * W + c0 + b] = acc[q][b];
                    }
                }
            }
        }
        const int c0 = c_end;

        // Leftover columns, one pixel at a time: the left edge [0, R), then
        // everything after the last full tile. min/max keep this right when
        // W < 2R + TILE_COLS.
        for (int r = r0; r < r0 + rows; ++r) {
            for (int c = 0; c < std::min(R, W); ++c) {
                conv_pixel(in, out, k, C, H, W, K, r, c);
            }
            for (int c = std::max(c0, R); c < W; ++c) {
                conv_pixel(in, out, k, C, H, W, K, r, c);
            }
        }
    }
}
