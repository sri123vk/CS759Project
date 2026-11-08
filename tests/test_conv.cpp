// Correctness tests for every conv2d implementation.
//
// Usage: test_conv [--impl SUBSTR] [--list] [-v]
//
//   --impl SUBSTR  only test implementations whose name contains SUBSTR
//   --list         print the implementations that would be tested and exit
//   -v             also print passing checks
//
// Every test runs on every implementation compiled in (see all_impls()):
//   seq                           always
//   omp/<schedule>/t<threads>     when built with -DCONV_HAVE_OMP
//   cuda                          when built with -DCONV_HAVE_CUDA
//
// Exit status: 0 if every check passes, 1 if any fails, 2 on bad usage.

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "conv2d.h"
#include "utils.h"

#ifdef CONV_HAVE_OMP
#include <omp.h>
#endif

namespace {

using ConvFn = void (*)(const float*, float*, const float*, int, int, int);
using Image = std::vector<float>;

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
};

std::vector<Impl> all_impls() {
    std::vector<Impl> v;
    v.push_back({"seq", conv2d_seq, [] {}, false});
#ifdef CONV_HAVE_OMP
    // Each pixel is still summed by one thread in seq's order, so the
    // result must be identical to seq, not merely close. An odd thread
    // count gives uneven row splits; dynamic scheduling shuffles which
    // thread gets which rows.
    std::vector<int> threads = {1, 2, 3};
    const int max_threads = omp_get_max_threads();
    if (max_threads > 3) threads.push_back(max_threads);
    const struct { const char* name; omp_sched_t kind; } scheds[] = {
        {"static", omp_sched_static}, {"dynamic", omp_sched_dynamic}};
    for (const auto& s : scheds) {
        for (int t : threads) {
            const omp_sched_t kind = s.kind;
            v.push_back({fmt("omp/%s/t%d", s.name, t), conv2d_omp,
                         [t, kind] { omp_set_num_threads(t); omp_set_schedule(kind, 0); },
                         true});
        }
    }
#endif
#ifdef CONV_HAVE_CUDA
    // FMA and a different evaluation order, so only the error bound applies.
    v.push_back({"cuda", conv2d_cuda, [] {}, false});
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
std::string run(const Impl& im, const Image& in, const Image& k,
                int H, int W, int K, Image& out) {
    const size_t n = (size_t)H * W;
    Image buf(n + 2 * kGuard, kGuardVal);
    std::fill(buf.begin() + kGuard, buf.begin() + kGuard + n, kNaN);
    const Image in_before = in, k_before = k;

    im.setup();
    im.fn(in.data(), buf.data() + kGuard, k.data(), H, W, K);

    out.assign(buf.begin() + kGuard, buf.begin() + kGuard + n);
    for (size_t i = 0; i < kGuard; ++i) {
        if (buf[i] != kGuardVal) return "wrote before the start of the output buffer";
        if (buf[kGuard + n + i] != kGuardVal) return "wrote past the end of the output buffer";
    }
    if (std::memcmp(in.data(), in_before.data(), n * sizeof(float)) != 0)
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
// no shared code with the implementations.
Image reference(const Image& in, const Image& k, int H, int W, int K) {
    const int R = K / 2;
    Image out((size_t)H * W);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            double acc = 0.0;
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < K; ++j) {
                    const int y = r + i - R, x = c + j - R;
                    if (y >= 0 && y < H && x >= 0 && x < W)
                        acc += (double)k[i * K + j] * in[(size_t)y * W + x];
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

    void begin_group(const char* name) {
        end_group();
        group_ = name;
        std::fill(group_fails_.begin(), group_fails_.end(), 0);
        std::printf("== %s\n", name);
    }

    void record(size_t impl, const std::string& test, const std::string& err) {
        ++checks_[impl];
        if (err.empty()) {
            if (verbose_) std::printf("   ok    %-20s %s\n", impls_[impl].name.c_str(), test.c_str());
            return;
        }
        ++fails_[impl];
        // Cap the output so one broken implementation doesn't bury everything else.
        if (++group_fails_[impl] <= kMaxPrinted)
            std::printf("   FAIL  %-20s %s: %s\n", impls_[impl].name.c_str(),
                        test.c_str(), err.c_str());
    }

    // Every implementation must reproduce want exactly (value equality).
    void expect_exact(const std::string& test, const Image& in, const Image& k,
                      int H, int W, int K, const Image& want) {
        Image out;
        for (size_t i = 0; i < impls_.size(); ++i) {
            std::string err = run(impls_[i], in, k, H, W, K, out);
            if (err.empty()) err = first_mismatch(want, out, W, false);
            record(i, test, err);
        }
    }

    // Prints the summary; returns true if everything passed.
    bool finish() {
        end_group();
        std::printf("\n%-20s %8s %8s\n", "implementation", "checks", "failed");
        int total_fails = 0;
        for (size_t i = 0; i < impls_.size(); ++i) {
            std::printf("%-20s %8d %8d\n", impls_[i].name.c_str(), checks_[i], fails_[i]);
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
                std::printf("   ...   %-20s %d more failures in this group\n",
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
            t.expect_exact(fmt("%dx%d K=%d", H, W, K), in, k, H, W, K, in);
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
                                   in, k, H, W, K, want);
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
                               in, k, H, W, K, want);
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
            t.expect_exact(fmt("%dx%d K=%d", H, W, K), in, k, H, W, K, want);
        }
    }
}

// ---------------------------------------------------------------------------
// Random data: within the FP32 error bound of a double-precision reference
// (same criterion as the driver's --check), and bit-identical to seq where
// the implementation promises that.

void test_random(Tester& t) {
    t.begin_group("random data: within FP32 error bound of double reference");
    // Tiny, thin, non-square, odd, prime and not-a-multiple-of-block shapes.
    const int shapes[][2] = {{1, 1}, {1, 17}, {17, 1}, {2, 2}, {3, 3}, {5, 7},
                             {7, 5}, {31, 33}, {33, 65}, {64, 64}, {257, 263}, {1000, 1000}};
    for (const auto& s : shapes) {
        const int H = s[0], W = s[1];
        for (int K : {1, 3, 5, 7}) {
            Image in((size_t)H * W), k((size_t)K * K), seq_out((size_t)H * W);
            fill_random(in, seed_for(H, W, K));
            fill_random(k, seed_for(K, K, H));
            const Image ref = reference(in, k, H, W, K);
            const std::vector<double> scale = sum_abs_terms(in, k, H, W, K);
            conv2d_seq(in.data(), seq_out.data(), k.data(), H, W, K);

            const std::string test = fmt("%dx%d K=%d", H, W, K);
            Image out;
            for (size_t i = 0; i < t.impls().size(); ++i) {
                const Impl& im = t.impls()[i];
                std::string err = run(im, in, k, H, W, K, out);
                if (err.empty()) {
                    const ErrStats e = compare(ref, out, scale, K);
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
        std::string err = run(t.impls()[i], in, k, H, W, K, first);
        for (int rep = 1; rep < 3 && err.empty(); ++rep) {
            err = run(t.impls()[i], in, k, H, W, K, out);
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
    test_repeatable(t);
    return t.finish() ? 0 : 1;
}
