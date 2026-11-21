// Benchmark driver.
//
// Usage:
//   conv --impl NAME --size N [--K 3] [--C 1] [--iters 10] [--warmup 2]
//        [--block BX BY|auto] [--check] [--dump DIR] [--header]
//   CUDA variants include cuda_multiout_const, cuda_multiout_tiled, and
//   cuda_multiout_tiled_const, alongside the existing implementations.
//
// Prints one CSV row per run (use --header to print the column names first).
// --C      input channels (CHW layout; output is one channel). seq and every
//          CUDA implementation support C > 1; the OpenMP versions don't yet.
// --block  CUDA thread-block shape (default 32 8); auto tunes candidates
// --check  also runs the sequential version and compares against it
// --dump   writes in/k/out as raw float32 files for the Python validator
//
// Timing: median_ms/min_ms are host wall-clock times per call. For cuda a
// call is host->device copy + kernel + device->host copy. Device memory is
// allocated once before timing (reported as alloc_ms) and the CUDA context
// is created before that, so neither is in median_ms. h2d_ms, kernel_ms and
// d2h_ms are medians of per-stage cudaEvent times. CUDA-only columns are
// empty for CPU implementations.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

#include "conv2d.h"
#include "utils.h"

namespace {

struct Args {
    std::string impl = "seq";
    int H = 512, W = 512, K = 3;
    int C = 1;  // input channels
    int iters = 10, warmup = 2;
    int bx = 32, by = 8;
    bool auto_block = false;
    bool check = false, header = false;
    std::string dump_dir;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --impl NAME --size N [--H h --W w] [--K k] [--C c]\n"
        "          CPU:  seq omp omp_simd omp_blocked\n"
        "          CUDA: cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout\n"
        "                cuda_multiout_const cuda_multiout_tiled cuda_multiout_tiled_const\n"
        "          --C > 1 (multi-channel input): seq and all CUDA, not OpenMP\n"
        "          [--iters n] [--warmup n] [--block bx by|auto]\n"
        "          [--check] [--dump dir] [--header]\n",
        prog);
    std::exit(1);
}

// Implementations that take a channel count: everything except the OpenMP
// versions, which are single-channel.
bool supports_multichannel(const std::string& impl) {
    return impl.rfind("omp", 0) != 0;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* s) { return std::strcmp(argv[i], s) == 0; };
        auto next = [&]() { if (i + 1 >= argc) usage(argv[0]); return argv[++i]; };
        if (is("--impl")) a.impl = next();
        else if (is("--size")) a.H = a.W = std::atoi(next());
        else if (is("--H")) a.H = std::atoi(next());
        else if (is("--W")) a.W = std::atoi(next());
        else if (is("--K")) a.K = std::atoi(next());
        else if (is("--C")) a.C = std::atoi(next());
        else if (is("--iters")) a.iters = std::atoi(next());
        else if (is("--warmup")) a.warmup = std::atoi(next());
        else if (is("--block")) {
            const char* value = next();
            if (std::strcmp(value, "auto") == 0) {
                a.auto_block = true;
            } else {
                a.bx = std::atoi(value);
                a.by = std::atoi(next());
                a.auto_block = false;
            }
        }
        else if (is("--check")) a.check = true;
        else if (is("--header")) a.header = true;
        else if (is("--dump")) a.dump_dir = next();
        else usage(argv[0]);
    }
    if (a.K <= 0 || a.K % 2 == 0) {
        std::fprintf(stderr, "K must be a positive odd number\n");
        std::exit(1);
    }
    if (a.H <= 0 || a.W <= 0 || a.C <= 0 || a.iters <= 0 || a.warmup < 0 ||
        a.bx <= 0 || a.by <= 0)
        usage(argv[0]);
    if (a.C != 1 && !supports_multichannel(a.impl)) {
        std::fprintf(stderr, "impl %s supports only --C 1 (the OpenMP versions are single-channel)\n",
                     a.impl.c_str());
        std::exit(1);
    }
    return a;
}

// Names served by cuda_conv_run_variant. They are recognized even in CPU-only
// builds, so main can report a useful error if CUDA was not compiled in.
bool is_cuda_variant(const std::string& impl) {
    return impl == "cuda_opt" ||
           impl == "cuda_const" || impl == "cuda_tiled_const" ||
           impl == "cuda_multiout" || impl == "cuda_multiout_const" ||
           impl == "cuda_multiout_tiled" || impl == "cuda_multiout_tiled_const";
}

// CPU implementations as called by the benchmark: (in, out, k, C, H, W, K).
using ConvFn = void (*)(const float*, float*, const float*, int, int, int, int);
using SingleChannelFn = void (*)(const float*, float*, const float*, int, int, int);

// Adapts a single-channel implementation to ConvFn. parse() has already
// rejected C != 1 for these.
template <SingleChannelFn F>
void single_channel(const float* in, float* out, const float* k,
                    int /*C*/, int H, int W, int K) {
    F(in, out, k, H, W, K);
}

ConvFn pick_cpu(const std::string& impl) {
    if (impl == "seq") return conv2d_seq;
#ifdef CONV_HAVE_OMP
    if (impl == "omp") return single_channel<conv2d_omp>;
#endif
#ifdef CONV_HAVE_OMP_SIMD
    if (impl == "omp_simd") return single_channel<conv2d_omp_simd>;
#endif
#ifdef CONV_HAVE_OMP_BLOCKED
    if (impl == "omp_blocked") return single_channel<conv2d_omp_blocked>;
#endif
    std::fprintf(stderr, "unknown impl: %s\n", impl.c_str());
    std::exit(1);
}

const double kNone = std::numeric_limits<double>::quiet_NaN();

// Timings for one benchmark configuration. Stage times stay NaN (printed as
// empty CSV fields) for implementations that don't have them.
struct Timings {
    std::vector<double> call_ms;
    double h2d_ms = kNone, kernel_ms = kNone, d2h_ms = kNone, alloc_ms = kNone;
};

Timings bench_cpu(const Args& a, const std::vector<float>& in,
                  std::vector<float>& out, const std::vector<float>& k) {
    ConvFn fn = pick_cpu(a.impl);
    auto call = [&] { fn(in.data(), out.data(), k.data(), a.C, a.H, a.W, a.K); };
    for (int i = 0; i < a.warmup; ++i) call();
    Timings t;
    t.call_ms.reserve(a.iters);
    for (int i = 0; i < a.iters; ++i) t.call_ms.push_back(time_ms(call));
    return t;
}

#ifdef CONV_HAVE_CUDA
Timings bench_cuda(const Args& a, const std::vector<float>& in,
                   std::vector<float>& out, const std::vector<float>& k) {
    cuda_init();
    const int max_threads = cuda_max_threads_per_block();
    const long long threads = static_cast<long long>(a.bx) * a.by;
    if (threads > max_threads) {
        std::fprintf(stderr, "block %dx%d has %lld threads; this GPU allows at most %d\n",
                     a.bx, a.by, threads, max_threads);
        std::exit(1);
    }

    Timings t;
    CudaConv c;
    t.alloc_ms = time_ms([&] { c = cuda_conv_alloc(a.C, a.H, a.W, a.K); });
    for (int i = 0; i < a.warmup; ++i)
        cuda_conv_run(c, in.data(), out.data(), k.data(), a.bx, a.by);

    std::vector<double> h2d, kern, d2h;
    for (int i = 0; i < a.iters; ++i) {
        CudaTimes s;
        t.call_ms.push_back(time_ms([&] {
            s = cuda_conv_run(c, in.data(), out.data(), k.data(), a.bx, a.by);
        }));
        h2d.push_back(s.h2d_ms);
        kern.push_back(s.kernel_ms);
        d2h.push_back(s.d2h_ms);
    }
    cuda_conv_free(c);

    t.h2d_ms = median(h2d);
    t.kernel_ms = median(kern);
    t.d2h_ms = median(d2h);
    return t;
}
#endif
#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_VARIANTS)
// Map each variant's command-line name to its kernel.
CudaTimes run_variant(const std::string& impl, const CudaConv& c,
                      const float* in, float* out, const float* k,
                      int bx, int by) {
    if (impl == "cuda_opt")
        return cuda_conv_run_variant(c, in, out, k, bx, by, CudaVariant::Tiled);
    if (impl == "cuda_const")
        return cuda_conv_run_variant(c, in, out, k, bx, by, CudaVariant::Constant);
    if (impl == "cuda_tiled_const")
        return cuda_conv_run_variant(c, in, out, k, bx, by, CudaVariant::TiledConstant);
    if (impl == "cuda_multiout")
        return cuda_conv_run_variant(c, in, out, k, bx, by, CudaVariant::MultiOutput);
    if (impl == "cuda_multiout_const")
        return cuda_conv_run_variant(c, in, out, k, bx, by, CudaVariant::MultiOutputConstant);
    if (impl == "cuda_multiout_tiled")
        return cuda_conv_run_variant(c, in, out, k, bx, by, CudaVariant::MultiOutputTiled);
    if (impl == "cuda_multiout_tiled_const")
        return cuda_conv_run_variant(c, in, out, k, bx, by, CudaVariant::MultiOutputTiledConstant);
    throw std::invalid_argument("unknown CUDA variant: " + impl);
}

// Allocate once, then time the variant's copy/kernel/copy path like cuda.
Timings bench_cuda_variant(const Args& a, const std::vector<float>& in,
                           std::vector<float>& out, const std::vector<float>& k) {
    cuda_init();
    if (!cuda_block_supported(a.H, a.W, a.bx, a.by)) {
        std::fprintf(stderr, "unsupported block/grid configuration %dx%d\n", a.bx, a.by);
        std::exit(1);
    }
    Timings t;
    CudaConv c;
    t.alloc_ms = time_ms([&] { c = cuda_conv_alloc(a.C, a.H, a.W, a.K); });
    std::vector<double> h2d, kern, d2h;
    try {
        for (int i = 0; i < a.warmup; ++i)
            run_variant(a.impl, c, in.data(), out.data(), k.data(), a.bx, a.by);
        t.call_ms.reserve(a.iters);
        for (int i = 0; i < a.iters; ++i) {
            CudaTimes stages;
            t.call_ms.push_back(time_ms([&] {
                stages = run_variant(a.impl, c, in.data(), out.data(), k.data(), a.bx, a.by);
            }));
            h2d.push_back(stages.h2d_ms);
            kern.push_back(stages.kernel_ms);
            d2h.push_back(stages.d2h_ms);
        }
    } catch (...) {
        cuda_conv_free(c);
        throw;
    }
    cuda_conv_free(c);
    t.h2d_ms = median(h2d);
    t.kernel_ms = median(kern);
    t.d2h_ms = median(d2h);
    return t;
}

// Measure a small, explicit set of 2D blocks on the actual workload and GPU.
// Tune outside the reported benchmark samples; selection cost is sent to stderr.
void autotune_block(Args& a, const std::vector<float>& in,
                    std::vector<float>& out, const std::vector<float>& k) {
    const int candidates[][2] = {{32, 8}, {32, 16}, {16, 16}, {32, 4}, {16, 8}, {8, 8}};
    constexpr int trials = 3;
    int best_x = 0, best_y = 0;
    double best_kernel = std::numeric_limits<double>::infinity();
    cuda_init();
    const double tuning_ms = time_ms([&] {
        CudaConv c = cuda_conv_alloc(a.C, a.H, a.W, a.K);
        try {
            for (const auto& block : candidates) {
                const int bx = block[0], by = block[1];
                if (!cuda_block_supported(a.H, a.W, bx, by)) continue;
                auto run = [&]() {
                    return a.impl == "cuda"
                        ? cuda_conv_run(c, in.data(), out.data(), k.data(), bx, by)
                        : run_variant(a.impl, c, in.data(), out.data(), k.data(), bx, by);
                };
                try {
                    run(); // One warm-up for this candidate.
                    std::vector<double> samples;
                    for (int i = 0; i < trials; ++i) samples.push_back(run().kernel_ms);
                    const double kernel_ms = median(samples);
                    if (std::isfinite(kernel_ms) && kernel_ms >= 0 && kernel_ms < best_kernel) {
                        best_kernel = kernel_ms;
                        best_x = bx;
                        best_y = by;
                    }
                } catch (const std::invalid_argument& error) {
                    std::fprintf(stderr, "skip auto-block %dx%d: %s\n", bx, by, error.what());
                }
            }
        } catch (...) {
            cuda_conv_free(c);
            throw;
        }
        cuda_conv_free(c);
    });
    if (best_x == 0) {
        std::fprintf(stderr, "no supported auto-block candidate for %dx%d\n", a.H, a.W);
        std::exit(1);
    }
    a.bx = best_x;
    a.by = best_y;
    std::fprintf(stderr,
                 "auto-block %s %dx%d K=%d: %dx%d (trial kernel median %.6f ms; tuning %.3f ms, excluded from reported timings)\n",
                 a.impl.c_str(), a.H, a.W, a.K, a.bx, a.by, best_kernel, tuning_ms);
}
#endif
// "%.4f", or an empty field for NaN.
std::string field(double v) {
    if (std::isnan(v)) return "";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4f", v);
    return buf;
}

}  // namespace
int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    const bool cuda = (a.impl == "cuda");
    const bool cuda_variant = is_cuda_variant(a.impl);
    const size_t n = (size_t)a.H * a.W;
    // CHW: C input planes and C filter slices; one output plane.
    // out starts as NaN so any pixel an implementation never writes fails --check.
    std::vector<float> in((size_t)a.C * n), k((size_t)a.C * a.K * a.K),
        out(n, std::numeric_limits<float>::quiet_NaN());
    fill_random(in, 759);
    fill_random(k, 42);

    Timings t;
    try {
    if (a.auto_block) {
        if (!cuda && !cuda_variant) {
            std::fprintf(stderr, "--block auto requires a CUDA implementation\n");
            return 1;
        }
#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_VARIANTS)
        autotune_block(a, in, out, k);
#else
        std::fprintf(stderr, "--block auto requires a CUDA build\n");
        return 1;
#endif
    }

    if (cuda) {
#ifdef CONV_HAVE_CUDA
    t = bench_cuda(a, in, out, k);
#else
    std::fprintf(
        stderr,
        "built without CUDA (nvcc not found when running make)\n"
    );
    return 1;
#endif

    } else if (cuda_variant) {
#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_VARIANTS)
    t = bench_cuda_variant(a, in, out, k);
#else
    std::fprintf(stderr, "CUDA implementation '%s' is unavailable in this build\n", a.impl.c_str());
    return 1;
#endif
    } else {
    t = bench_cpu(a, in, out, k);
    }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA benchmark failed: %s\n", error.what());
        return 1;
    }

    ErrStats err;
    bool pass = true;
    if (a.check) {
        std::vector<float> ref(n);
        conv2d_seq(in.data(), ref.data(), k.data(), a.C, a.H, a.W, a.K);
        err = compare(ref, out, sum_abs_terms(in, k, a.C, a.H, a.W, a.K),
                      (long long)a.C * a.K * a.K);
        pass = within_tol(err);
    }

    if (!a.dump_dir.empty()) {
        // {H}x{W}_k{K}_ for one channel (unchanged), {H}x{W}_k{K}_c{C}_ otherwise.
        std::string base = a.dump_dir + "/" + std::to_string(a.H) + "x" +
                           std::to_string(a.W) + "_k" + std::to_string(a.K) + "_" +
                           (a.C != 1 ? "c" + std::to_string(a.C) + "_" : "");
        if (!write_bin(base + "in.bin", in) || !write_bin(base + "k.bin", k) ||
            !write_bin(base + a.impl + "_out.bin", out)) {
            std::fprintf(stderr, "failed to write dump files to %s\n", a.dump_dir.c_str());
            return 1;
        }
    }

    const std::string bx = (cuda || cuda_variant) ? std::to_string(a.bx) : "";
    const std::string by = (cuda || cuda_variant) ? std::to_string(a.by) : "";
    if (a.header)
        std::printf("impl,H,W,K,C,bx,by,iters,median_ms,min_ms,"
                    "h2d_ms,kernel_ms,d2h_ms,alloc_ms,"
                    "max_abs_err,max_rel_err,err_ratio,nonfinite,pass\n");
    std::printf("%s,%d,%d,%d,%d,%s,%s,%d,%.4f,%.4f,%s,%s,%s,%s,%.3e,%.3e,%.3e,%zu,%d\n",
                a.impl.c_str(), a.H, a.W, a.K, a.C, bx.c_str(), by.c_str(), a.iters,
                median(t.call_ms),
                *std::min_element(t.call_ms.begin(), t.call_ms.end()),
                field(t.h2d_ms).c_str(), field(t.kernel_ms).c_str(),
                field(t.d2h_ms).c_str(), field(t.alloc_ms).c_str(),
                err.max_abs, err.max_rel, err.max_ratio, err.nonfinite,
                pass ? 1 : 0);
    return pass ? 0 : 2;
}
