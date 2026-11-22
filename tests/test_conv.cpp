// Correctness tests for every conv2d implementation.
//
// Usage: test_conv [--impl SUBSTR] [--list] [-v]
//
//   --impl SUBSTR  only test implementations whose name contains SUBSTR
//   --list         print the implementations that would be tested and exit
//   -v             also print passing checks
//
// Every test runs on every implementation compiled in (see all_impls()):
//   seq                                   always
//   omp/<schedule>/t<threads>             when built with -DCONV_HAVE_OMP
//   omp_simd/<schedule>/t<threads>        when built with -DCONV_HAVE_OMP_SIMD
//   omp_blocked/<schedule>/t<threads>     when built with -DCONV_HAVE_OMP_BLOCKED
//   cuda/<bx>x<by>                        when built with -DCONV_HAVE_CUDA
//   cuda_opt, cuda_const, cuda_tiled_const, cuda_multiout and the multiout
//   combinations /<bx>x<by>
//                                         when built with -DCONV_HAVE_CUDA_VARIANTS
//
// Most tests use one input channel. The multi-channel tests (C > 1) run only
// on implementations marked `multichannel` in all_impls(): seq and every CUDA
// implementation (not the OpenMP versions).
//
// Exit status: 0 if every check passes, 1 if any fails, 2 on bad usage.

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "conv2d.h"
#include "utils.h"

#if defined(CONV_HAVE_OMP) || defined(CONV_HAVE_OMP_SIMD) || defined(CONV_HAVE_OMP_BLOCKED)
#define CONV_HAVE_ANY_OMP
#include <omp.h>
#endif

namespace {

// (in, out, k, C, H, W, K), CHW layout.
using ConvFn = std::function<void(const float*, float*, const float*, int, int, int, int)>;
using SingleChannelFn = std::function<void(const float*, float*, const float*, int, int, int)>;
using Image = std::vector<float>;

// Adapts a single-channel implementation. It is only called with C == 1.
// Unused in a build with neither OpenMP nor CUDA.
[[maybe_unused]] ConvFn single_channel(SingleChannelFn f) {
    return [f](const float* in, float* out, const float* k, int /*C*/, int H, int W, int K) {
        f(in, out, k, H, W, K);
    };
}

const float kNaN = std::numeric_limits<float>::quiet_NaN();

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// ---------------------------------------------------------------------------
// Implementations under test

struct Impl {
    std::string name;
    ConvFn fn;
    std::function<void()> setup;  // runs before every call (e.g. OpenMP settings)
    bool bitwise_vs_seq;          // output must match conv2d_seq bit for bit
    bool multichannel = false;    // accepts C > 1; otherwise only tested with C == 1
};

std::vector<Impl> all_impls() {
    std::vector<Impl> v;
    v.push_back({"seq", conv2d_seq, [] {}, false, true});
#ifdef CONV_HAVE_ANY_OMP
    // Every OpenMP version runs under every schedule and thread count. An odd
    // thread count gives uneven row splits; dynamic and guided scheduling
    // shuffle which thread gets which rows.
    //
    // In every version each pixel is summed by one thread in seq's order, so
    // the result must be identical to seq, not merely close. The exception:
    // when FMA instructions are enabled (e.g. -march=native), GCC fuses the
    // vectorized loops of omp_simd and omp_blocked differently from seq's
    // loop, so those two then only have to meet the error bound.
#ifdef __FMA__
    const bool vector_bitwise = false;
#else
    const bool vector_bitwise = true;
#endif
    const struct { const char* name; SingleChannelFn fn; bool bitwise; } omps[] = {
#ifdef CONV_HAVE_OMP
        {"omp", conv2d_omp, true},
#endif
#ifdef CONV_HAVE_OMP_SIMD
        {"omp_simd", conv2d_omp_simd, vector_bitwise},
#endif
#ifdef CONV_HAVE_OMP_BLOCKED
        {"omp_blocked", conv2d_omp_blocked, vector_bitwise},
#endif
    };
    (void)vector_bitwise;  // unused when only conv2d_omp is built

    std::vector<int> threads = {1, 2, 3};
    const int max_threads = omp_get_max_threads();
    if (max_threads > 3) threads.push_back(max_threads);
    const struct { const char* name; omp_sched_t kind; } scheds[] = {
        {"static", omp_sched_static}, {"dynamic", omp_sched_dynamic},
        {"guided", omp_sched_guided}};
    for (const auto& o : omps) {
        for (const auto& s : scheds) {
            for (int t : threads) {
                const omp_sched_t kind = s.kind;
                v.push_back({fmt("%s/%s/t%d", o.name, s.name, t), single_channel(o.fn),
                             [t, kind] { omp_set_num_threads(t); omp_set_schedule(kind, 0); },
                             o.bitwise});
            }
        }
    }
#endif
#ifdef CONV_HAVE_CUDA
    // FMA changes the rounding, so only the error bound applies.
    // cuda/32x8 is the one-call conv2d_cuda(); the rest go through the
    // benchmark path (alloc / run / free) with shapes from the block sweep:
    // square, wider than a warp (64x4 to 1024x1), taller than wide (1xN,
    // 4x256), plus 7x3 and 33x5, which leave partial blocks along every edge.
    // All of them take C input channels.
    v.push_back({"cuda/32x8", conv2d_cuda, [] {}, false, true});
    const int blocks[][2] = {{8, 8}, {16, 16}, {32, 32}, {64, 4}, {64, 16}, {128, 1},
                             {256, 4}, {1024, 1}, {1, 256}, {4, 256}, {7, 3}, {33, 5}};
    for (const auto& b : blocks) {
        const int bx = b[0], by = b[1];
        v.push_back({fmt("cuda/%dx%d", bx, by),
                     [bx, by](const float* in, float* out, const float* k,
                              int C, int H, int W, int K) {
                         CudaConv c = cuda_conv_alloc(C, H, W, K);
                         cuda_conv_run(c, in, out, k, bx, by);
                         cuda_conv_free(c);
                     },
                     [] {}, false, true});
    }
#endif

#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_VARIANTS)
    // Test each variant with the default block, blocks wider than a warp
    // (64x4, 128x8 = 1024 threads, 256x1), a tall one (1x128), and odd
    // blocks that leave partial blocks at image boundaries (7x3, 33x5).
    // For the shared-tile variants the wide blocks also mean wide tiles; all
    // of these fit in 48 KB of shared memory for K <= 11.
    const struct { const char* name; CudaVariant variant; } extra_variants[] = {
        {"cuda_opt", CudaVariant::Tiled},
        {"cuda_const", CudaVariant::Constant},
        {"cuda_tiled_const", CudaVariant::TiledConstant},
        {"cuda_multiout", CudaVariant::MultiOutput},
        {"cuda_multiout_const", CudaVariant::MultiOutputConstant},
        {"cuda_multiout_tiled", CudaVariant::MultiOutputTiled},
        {"cuda_multiout_tiled_const", CudaVariant::MultiOutputTiledConstant}
    };
    const int extra_blocks[][2] = {{32, 8}, {64, 4}, {128, 8}, {256, 1}, {1, 128},
                                   {7, 3}, {33, 5}};
    for (const auto& item : extra_variants) {
        for (const auto& block : extra_blocks) {
            const int bx = block[0], by = block[1];
            const CudaVariant variant = item.variant;
            v.push_back({fmt("%s/%dx%d", item.name, bx, by),
                [bx, by, variant](const float* in, float* out, const float* k,
                                  int C, int H, int W, int K) {
                    CudaConv c = cuda_conv_alloc(C, H, W, K);
                    try {
                        cuda_conv_run_variant(c, in, out, k, bx, by, variant);
                    } catch (...) {
                        cuda_conv_free(c);
                        throw;
                    }
                    cuda_conv_free(c);
                },
                [] {}, false, true});
        }
    }
#endif

    return v;
}

// ---------------------------------------------------------------------------
// Running an implementation and checking the result

constexpr size_t kGuard = 64;
constexpr float kGuardVal = -12345.0f;

// Runs one implementation. The output starts as NaN, so unwritten pixels
// fail later comparisons, and is surrounded by guard values, so writes
// past either end are caught. Returns "" or a description of what went wrong.
// `in` holds C planes of H*W and `k` holds C slices of K*K (CHW).
std::string run(const Impl& im, const Image& in, const Image& k,
                int C, int H, int W, int K, Image& out) {
    const size_t n = (size_t)H * W;
    if (in.size() != (size_t)C * n || k.size() != (size_t)C * K * K)
        return "test bug: input or filter size does not match C, H, W, K";
    Image buf(n + 2 * kGuard, kGuardVal);
    std::fill(buf.begin() + kGuard, buf.begin() + kGuard + n, kNaN);
    const Image in_before = in, k_before = k;

    im.setup();
    // A CUDA launcher that rejects a configuration throws; report that as a
    // failure of this check instead of aborting the whole run.
    try {
        im.fn(in.data(), buf.data() + kGuard, k.data(), C, H, W, K);
    } catch (const std::exception& e) {
        return std::string("threw: ") + e.what();
    }

    out.assign(buf.begin() + kGuard, buf.begin() + kGuard + n);
    for (size_t i = 0; i < kGuard; ++i) {
        if (buf[i] != kGuardVal) return "wrote before the start of the output buffer";
        if (buf[kGuard + n + i] != kGuardVal) return "wrote past the end of the output buffer";
    }
    if (std::memcmp(in.data(), in_before.data(), in.size() * sizeof(float)) != 0)
        return "modified the input image";
    if (std::memcmp(k.data(), k_before.data(), k.size() * sizeof(float)) != 0)
        return "modified the filter";
    return "";
}

// "" if got matches want everywhere, else the first mismatch.
// bitwise=false compares values (so 0 == -0); NaN never matches either way.
std::string first_mismatch(const Image& want, const Image& got, int W, bool bitwise) {
    for (size_t i = 0; i < want.size(); ++i) {
        bool same;
        if (bitwise) {
            uint32_t a, b;
            std::memcpy(&a, &want[i], 4);
            std::memcpy(&b, &got[i], 4);
            same = a == b;
        } else {
            same = got[i] == want[i];
        }
        if (!same)
            return fmt("at (r=%zu, c=%zu): got %.9g, want %.9g",
                       i / W, i % W, got[i], want[i]);
    }
    return "";
}

// Straightforward reference: double accumulation, explicit bounds checks,
// no shared code with the implementations. CHW input, C channels summed.
Image reference(const Image& in, const Image& k, int C, int H, int W, int K) {
    const int R = K / 2;
    Image out((size_t)H * W);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            double acc = 0.0;
            for (int ch = 0; ch < C; ++ch)
                for (int i = 0; i < K; ++i)
                    for (int j = 0; j < K; ++j) {
                        const int y = r + i - R, x = c + j - R;
                        if (y >= 0 && y < H && x >= 0 && x < W)
                            acc += (double)k[((size_t)ch * K + i) * K + j] *
                                   in[((size_t)ch * H + y) * W + x];
                    }
            out[(size_t)r * W + c] = (float)acc;
        }
    return out;
}

// ---------------------------------------------------------------------------
// Bookkeeping

class Tester {
public:
    Tester(std::vector<Impl> impls, bool verbose)
        : impls_(std::move(impls)), verbose_(verbose),
          checks_(impls_.size()), fails_(impls_.size()), group_fails_(impls_.size()) {}

    const std::vector<Impl>& impls() const { return impls_; }

    // Whether impl i can be run with C input channels.
    bool runs_with(size_t i, int C) const { return C == 1 || impls_[i].multichannel; }

    void begin_group(const char* name) {
        end_group();
        group_ = name;
        std::fill(group_fails_.begin(), group_fails_.end(), 0);
        std::printf("== %s\n", name);
    }

    void record(size_t impl, const std::string& test, const std::string& err) {
        ++checks_[impl];
        if (err.empty()) {
            if (verbose_) std::printf("   ok    %-24s %s\n", impls_[impl].name.c_str(), test.c_str());
            return;
        }
        ++fails_[impl];
        // Cap the output so one broken implementation doesn't bury everything else.
        if (++group_fails_[impl] <= kMaxPrinted)
            std::printf("   FAIL  %-24s %s: %s\n", impls_[impl].name.c_str(),
                        test.c_str(), err.c_str());
    }

    // Every implementation that supports C channels must reproduce want
    // exactly (value equality).
    void expect_exact(const std::string& test, const Image& in, const Image& k,
                      int C, int H, int W, int K, const Image& want) {
        Image out;
        for (size_t i = 0; i < impls_.size(); ++i) {
            if (!runs_with(i, C)) continue;
            std::string err = run(impls_[i], in, k, C, H, W, K, out);
            if (err.empty()) err = first_mismatch(want, out, W, false);
            record(i, test, err);
        }
    }

    // Prints the summary; returns true if everything passed.
    bool finish() {
        end_group();
        std::printf("\n%-24s %8s %8s\n", "implementation", "checks", "failed");
        int total_fails = 0;
        for (size_t i = 0; i < impls_.size(); ++i) {
            std::printf("%-24s %8d %8d\n", impls_[i].name.c_str(), checks_[i], fails_[i]);
            total_fails += fails_[i];
        }
        std::printf("\n%s\n", total_fails ? "FAILED" : "ALL PASSED");
        return total_fails == 0;
    }

private:
    static constexpr int kMaxPrinted = 5;

    void end_group() {
        if (group_.empty()) return;
        for (size_t i = 0; i < impls_.size(); ++i)
            if (group_fails_[i] > kMaxPrinted)
                std::printf("   ...   %-24s %d more failures in this group\n",
                            impls_[i].name.c_str(), group_fails_[i] - kMaxPrinted);
        group_.clear();
    }

    std::vector<Impl> impls_;
    bool verbose_;
    std::vector<int> checks_, fails_, group_fails_;
    std::string group_;
};

// ---------------------------------------------------------------------------
// Tests with exact expected answers. Every value involved is either a small
// integer or a product with 1 or 0, so the result is exact in FP32 for any
// summation order, with or without FMA.

unsigned seed_for(int H, int W, int K) { return 1000003u * H + 1009u * W + K; }

// Center tap = 1: output must equal the input.
void test_identity(Tester& t) {
    t.begin_group("identity kernel: output == input");
    const int shapes[][2] = {{1, 1}, {5, 7}, {33, 65}};
    for (const auto& s : shapes) {
        const int H = s[0], W = s[1];
        for (int K : {1, 3, 5, 7}) {
            Image in((size_t)H * W), k((size_t)K * K, 0.0f);
            fill_random(in, seed_for(H, W, K));
            k[(K / 2) * K + K / 2] = 1.0f;
            t.expect_exact(fmt("%dx%d K=%d", H, W, K), in, k, 1, H, W, K, in);
        }
    }
}

// One tap = 1 anywhere in the filter: output is the input shifted, with
// zeros shifted in. Non-square images expose swapped row/column indexing.
void test_shift(Tester& t) {
    t.begin_group("single-tap kernel: shifted input with zero fill");
    const int shapes[][2] = {{9, 13}, {2, 3}};
    for (const auto& s : shapes) {
        const int H = s[0], W = s[1];
        for (int K : {3, 5, 7}) {
            const int R = K / 2;
            Image in((size_t)H * W);
            fill_random(in, seed_for(H, W, K));
            for (int i0 = 0; i0 < K; ++i0)
                for (int j0 = 0; j0 < K; ++j0) {
                    Image k((size_t)K * K, 0.0f), want((size_t)H * W, 0.0f);
                    k[i0 * K + j0] = 1.0f;
                    for (int r = 0; r < H; ++r)
                        for (int c = 0; c < W; ++c) {
                            const int y = r + i0 - R, x = c + j0 - R;
                            if (y >= 0 && y < H && x >= 0 && x < W)
                                want[(size_t)r * W + c] = in[(size_t)y * W + x];
                        }
                    t.expect_exact(fmt("%dx%d K=%d tap=(%d,%d)", H, W, K, i0, j0),
                                   in, k, 1, H, W, K, want);
                }
        }
    }
}

// Single 1 in the image: the output around it is the filter rotated 180
// degrees. This is what distinguishes cross-correlation (our spec) from
// true convolution. Impulses at corners also exercise boundary clipping.
void test_impulse(Tester& t) {
    t.begin_group("impulse image: output is the filter rotated 180 degrees");
    const int shapes[][2] = {{15, 11}, {4, 6}};
    for (const auto& s : shapes) {
        const int H = s[0], W = s[1];
        const int points[][2] = {{H / 2, W / 2}, {0, 0}, {0, W - 1}, {H - 1, 0}, {H - 1, W - 1}};
        for (int K : {3, 5, 7}) {
            const int R = K / 2;
            Image k((size_t)K * K);
            fill_random(k, seed_for(H, W, K));
            for (const auto& p : points) {
                const int py = p[0], px = p[1];
                Image in((size_t)H * W, 0.0f), want((size_t)H * W, 0.0f);
                in[(size_t)py * W + px] = 1.0f;
                for (int r = 0; r < H; ++r)
                    for (int c = 0; c < W; ++c) {
                        const int i = py - r + R, j = px - c + R;
                        if (i >= 0 && i < K && j >= 0 && j < K)
                            want[(size_t)r * W + c] = k[i * K + j];
                    }
                t.expect_exact(fmt("%dx%d K=%d impulse=(%d,%d)", H, W, K, py, px),
                               in, k, 1, H, W, K, want);
            }
        }
    }
}

// All-ones image and filter: each output counts the filter taps that land
// inside the image, so zero padding shows up as smaller values at the edges.
// Includes images smaller than the filter.
void test_ones(Tester& t) {
    t.begin_group("all-ones image and filter: in-bounds tap counts");
    const int shapes[][2] = {{1, 1}, {2, 2}, {3, 3}, {4, 9}, {9, 4}, {16, 16}, {33, 65}};
    for (const auto& s : shapes) {
        const int H = s[0], W = s[1];
        for (int K : {1, 3, 5, 7}) {
            const int R = K / 2;
            Image in((size_t)H * W, 1.0f), k((size_t)K * K, 1.0f), want((size_t)H * W);
            for (int r = 0; r < H; ++r)
                for (int c = 0; c < W; ++c) {
                    const int rows = std::min(r + R, H - 1) - std::max(r - R, 0) + 1;
                    const int cols = std::min(c + R, W - 1) - std::max(c - R, 0) + 1;
                    want[(size_t)r * W + c] = (float)(rows * cols);
                }
            t.expect_exact(fmt("%dx%d K=%d", H, W, K), in, k, 1, H, W, K, want);
        }
    }
}

// ---------------------------------------------------------------------------
// Random data: within the FP32 error bound of a double-precision reference
// (same criterion as the driver's --check), and bit-identical to seq where
// the implementation promises that.

// One random C-channel problem, checked on every implementation that
// supports C channels.
void check_random(Tester& t, int C, int H, int W, int K, const std::string& test) {
    Image in((size_t)C * H * W), k((size_t)C * K * K), seq_out((size_t)H * W);
    fill_random(in, seed_for(H, W, K) + 7u * (C - 1));
    fill_random(k, seed_for(K, K, H) + 7u * (C - 1));
    const Image ref = reference(in, k, C, H, W, K);
    const std::vector<double> scale = sum_abs_terms(in, k, C, H, W, K);
    conv2d_seq(in.data(), seq_out.data(), k.data(), C, H, W, K);

    Image out;
    for (size_t i = 0; i < t.impls().size(); ++i) {
        if (!t.runs_with(i, C)) continue;
        const Impl& im = t.impls()[i];
        std::string err = run(im, in, k, C, H, W, K, out);
        if (err.empty()) {
            const ErrStats e = compare(ref, out, scale, (long long)C * K * K);
            if (!within_tol(e))
                err = fmt("err_ratio=%.3g (must be <= 1), nonfinite=%zu, max_abs=%.3g",
                          e.max_ratio, e.nonfinite, e.max_abs);
        }
        if (err.empty() && im.bitwise_vs_seq) {
            err = first_mismatch(seq_out, out, W, true);
            if (!err.empty()) err = "not bit-identical to seq " + err;
        }
        t.record(i, test, err);
    }
}

void test_random(Tester& t) {
    t.begin_group("random data: within FP32 error bound of double reference");
    // Tiny, thin, non-square, odd, prime and not-a-multiple-of-block shapes.
    const int shapes[][2] = {{1, 1}, {1, 17}, {17, 1}, {2, 2}, {3, 3}, {5, 7},
                             {7, 5}, {31, 33}, {33, 65}, {64, 64}, {257, 263}, {1000, 1000}};
    for (const auto& s : shapes)
        for (int K : {1, 3, 5, 7})
            check_random(t, 1, s[0], s[1], K, fmt("%dx%d K=%d", s[0], s[1], K));
}

// ---------------------------------------------------------------------------
// Multi-channel input (C > 1, CHW). Run only on implementations marked
// multichannel in all_impls().

// Exact cases, built so that every output is a single product with 1 or a
// sum of small integers, so any summation order gives the same result.
void test_multichannel_exact(Tester& t) {
    t.begin_group("multi-channel: channel selection, impulse, all-ones");
    const int shapes[][2] = {{9, 13}, {4, 6}, {1, 1}};
    for (const auto& s : shapes) {
        const int H = s[0], W = s[1];
        const size_t plane = (size_t)H * W;
        for (int C : {2, 3, 5}) {
            for (int K : {1, 3, 5}) {
                const int R = K / 2;
                const size_t slice = (size_t)K * K;

                // Selection: only filter slice ch0 is nonzero (center tap = 1),
                // so the output must be input channel ch0. Catches a wrong
                // input-channel offset.
                Image in((size_t)C * plane);
                fill_random(in, seed_for(H, W, K) + C);
                for (int ch0 = 0; ch0 < C; ++ch0) {
                    Image k((size_t)C * slice, 0.0f);
                    k[ch0 * slice + (size_t)R * K + R] = 1.0f;
                    const Image want(in.begin() + ch0 * plane, in.begin() + (ch0 + 1) * plane);
                    t.expect_exact(fmt("select %dx%d C=%d K=%d ch=%d", H, W, C, K, ch0),
                                   in, k, C, H, W, K, want);
                }

                // Impulse in channel ch0 only: the output around it is filter
                // slice ch0 rotated 180 degrees. Catches pairing an input
                // channel with the wrong filter slice.
                Image k((size_t)C * slice);
                fill_random(k, seed_for(K, C, H));
                const int py = H / 2, px = W / 2;
                for (int ch0 = 0; ch0 < C; ++ch0) {
                    Image imp((size_t)C * plane, 0.0f), want(plane, 0.0f);
                    imp[ch0 * plane + (size_t)py * W + px] = 1.0f;
                    for (int r = 0; r < H; ++r)
                        for (int c = 0; c < W; ++c) {
                            const int i = py - r + R, j = px - c + R;
                            if (i >= 0 && i < K && j >= 0 && j < K)
                                want[(size_t)r * W + c] = k[ch0 * slice + (size_t)i * K + j];
                        }
                    t.expect_exact(fmt("impulse %dx%d C=%d K=%d ch=%d", H, W, C, K, ch0),
                                   imp, k, C, H, W, K, want);
                }

                // All ones: C times the in-bounds tap count. Catches skipped
                // or repeated channels.
                Image ones_in((size_t)C * plane, 1.0f), ones_k((size_t)C * slice, 1.0f), want(plane);
                for (int r = 0; r < H; ++r)
                    for (int c = 0; c < W; ++c) {
                        const int rows = std::min(r + R, H - 1) - std::max(r - R, 0) + 1;
                        const int cols = std::min(c + R, W - 1) - std::max(c - R, 0) + 1;
                        want[(size_t)r * W + c] = (float)(C * rows * cols);
                    }
                t.expect_exact(fmt("ones %dx%d C=%d K=%d", H, W, C, K),
                               ones_in, ones_k, C, H, W, K, want);
            }
        }
    }
}

void test_multichannel_random(Tester& t) {
    t.begin_group("multi-channel random data: within FP32 error bound of double reference");
    const int shapes[][2] = {{1, 1}, {1, 17}, {5, 7}, {31, 33}, {64, 64}, {257, 263}};
    for (const auto& s : shapes)
        for (int C : {2, 3, 16})
            for (int K : {1, 3, 5, 7})
                check_random(t, C, s[0], s[1], K, fmt("%dx%d C=%d K=%d", s[0], s[1], C, K));
}

// Channel counts and filter sizes beyond the random group above: many
// channels, large filters, and C*K*K up to the 4096-float constant-memory
// limit (83*7*7 = 4067), which the constant-memory variants must accept.
void test_multichannel_configs(Tester& t) {
    t.begin_group("multi-channel configurations: many channels, large filters");
    const int shapes[][2] = {{1, 1}, {17, 1}, {33, 65}, {128, 128}};
    const int configs[][2] = {{4, 9}, {8, 11}, {32, 11}, {64, 7}, {83, 7}, {128, 5}};  // {C, K}
    for (const auto& s : shapes)
        for (const auto& ck : configs)
            check_random(t, ck[0], s[0], s[1], ck[1],
                         fmt("%dx%d C=%d K=%d", s[0], s[1], ck[0], ck[1]));
}

// Same multi-channel input three times must give the same bits. The shared-
// tile kernels reload their tile once per channel, so a missing barrier
// between channels shows up here as run-to-run differences.
void test_multichannel_repeatable(Tester& t) {
    t.begin_group("multi-channel repeatability: identical bits across 3 runs");
    const int C = 8, H = 257, W = 263, K = 5;
    Image in((size_t)C * H * W), k((size_t)C * K * K);
    fill_random(in, seed_for(H, W, K) + C);
    fill_random(k, seed_for(K, K, H) + C);
    for (size_t i = 0; i < t.impls().size(); ++i) {
        if (!t.runs_with(i, C)) continue;
        Image first, out;
        std::string err = run(t.impls()[i], in, k, C, H, W, K, first);
        for (int rep = 1; rep < 3 && err.empty(); ++rep) {
            err = run(t.impls()[i], in, k, C, H, W, K, out);
            if (err.empty()) {
                err = first_mismatch(first, out, W, true);
                if (!err.empty()) err = fmt("run %d differs from run 0 ", rep) + err;
            }
        }
        t.record(i, fmt("%dx%d C=%d K=%d", H, W, C, K), err);
    }
}

// Same input three times must give the same bits. Catches data races and
// reads of uninitialized memory that happen to land within tolerance.
void test_repeatable(Tester& t) {
    t.begin_group("repeatability: identical bits across 3 runs");
    const int H = 513, W = 517, K = 7;
    Image in((size_t)H * W), k((size_t)K * K);
    fill_random(in, seed_for(H, W, K));
    fill_random(k, seed_for(K, K, H));
    for (size_t i = 0; i < t.impls().size(); ++i) {
        Image first, out;
        std::string err = run(t.impls()[i], in, k, 1, H, W, K, first);
        for (int rep = 1; rep < 3 && err.empty(); ++rep) {
            err = run(t.impls()[i], in, k, 1, H, W, K, out);
            if (err.empty()) {
                err = first_mismatch(first, out, W, true);
                if (!err.empty()) err = fmt("run %d differs from run 0 ", rep) + err;
            }
        }
        t.record(i, fmt("%dx%d K=%d", H, W, K), err);
    }
}

void usage(const char* prog) {
    std::fprintf(stderr, "usage: %s [--impl SUBSTR] [--list] [-v]\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    std::string filter;
    bool list = false, verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--impl") && i + 1 < argc) filter = argv[++i];
        else if (!std::strcmp(argv[i], "--list")) list = true;
        else if (!std::strcmp(argv[i], "-v")) verbose = true;
        else usage(argv[0]);
    }

    std::vector<Impl> impls;
    for (auto& im : all_impls())
        if (im.name.find(filter) != std::string::npos) impls.push_back(std::move(im));
    if (impls.empty()) {
        std::fprintf(stderr, "no implementation matches '%s' (try --list)\n", filter.c_str());
        return 2;
    }
    if (list) {
        for (const auto& im : impls) std::printf("%s\n", im.name.c_str());
        return 0;
    }

    Tester t(std::move(impls), verbose);
    test_identity(t);
    test_shift(t);
    test_impulse(t);
    test_ones(t);
    test_random(t);
    test_multichannel_exact(t);
    test_multichannel_random(t);
    test_multichannel_configs(t);
    test_multichannel_repeatable(t);
    test_repeatable(t);
    return t.finish() ? 0 : 1;
}
