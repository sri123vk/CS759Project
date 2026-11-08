// Benchmark driver.
//
// Usage:
//   conv --impl seq|omp|cuda --size N [--K 3] [--iters 10] [--warmup 2]
//        [--block BX BY] [--check] [--dump DIR] [--header]
//
// Prints one CSV row per run (use --header to print the column names first).
// --block  CUDA thread-block shape (default 32 8)
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
#include <limits>
#include <string>
#include <vector>

#include "conv2d.h"
#include "utils.h"

namespace {

struct Args {
    std::string impl = "seq";
    int H = 512, W = 512, K = 3;
    int iters = 10, warmup = 2;
    int bx = 32, by = 8;
    bool check = false, header = false;
    std::string dump_dir;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --impl seq|omp|cuda --size N [--H h --W w] [--K k]\n"
        "          [--iters n] [--warmup n] [--block bx by]\n"
        "          [--check] [--dump dir] [--header]\n",
        prog);
    std::exit(1);
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
        else if (is("--iters")) a.iters = std::atoi(next());
        else if (is("--warmup")) a.warmup = std::atoi(next());
        else if (is("--block")) { a.bx = std::atoi(next()); a.by = std::atoi(next()); }
        else if (is("--check")) a.check = true;
        else if (is("--header")) a.header = true;
        else if (is("--dump")) a.dump_dir = next();
        else usage(argv[0]);
    }
    if (a.K <= 0 || a.K % 2 == 0) {
        std::fprintf(stderr, "K must be a positive odd number\n");
        std::exit(1);
    }
    if (a.H <= 0 || a.W <= 0 || a.iters <= 0 || a.warmup < 0 || a.bx <= 0 || a.by <= 0)
        usage(argv[0]);
    return a;
}

using ConvFn = void (*)(const float*, float*, const float*, int, int, int);

ConvFn pick_cpu(const std::string& impl) {
    if (impl == "seq") return conv2d_seq;
#ifdef CONV_HAVE_OMP
    if (impl == "omp") return conv2d_omp;
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
    for (int i = 0; i < a.warmup; ++i) fn(in.data(), out.data(), k.data(), a.H, a.W, a.K);
    Timings t;
    t.call_ms.reserve(a.iters);
    for (int i = 0; i < a.iters; ++i)
        t.call_ms.push_back(time_ms([&] { fn(in.data(), out.data(), k.data(), a.H, a.W, a.K); }));
    return t;
}

#ifdef CONV_HAVE_CUDA
Timings bench_cuda(const Args& a, const std::vector<float>& in,
                   std::vector<float>& out, const std::vector<float>& k) {
    cuda_init();
    const int max_threads = cuda_max_threads_per_block();
    if (a.bx * a.by > max_threads) {
        std::fprintf(stderr, "block %dx%d has %d threads; this GPU allows at most %d\n",
                     a.bx, a.by, a.bx * a.by, max_threads);
        std::exit(1);
    }

    Timings t;
    CudaConv c;
    t.alloc_ms = time_ms([&] { c = cuda_conv_alloc(a.H, a.W, a.K); });
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
    const bool cuda = a.impl == "cuda";

    const size_t n = (size_t)a.H * a.W;
    // out starts as NaN so any pixel an implementation never writes fails --check.
    std::vector<float> in(n), k((size_t)a.K * a.K),
        out(n, std::numeric_limits<float>::quiet_NaN());
    fill_random(in, 759);
    fill_random(k, 42);

    Timings t;
    if (cuda) {
#ifdef CONV_HAVE_CUDA
        t = bench_cuda(a, in, out, k);
#else
        std::fprintf(stderr, "built without CUDA (nvcc not found when running make)\n");
        return 1;
#endif
    } else {
        t = bench_cpu(a, in, out, k);
    }

    ErrStats err;
    bool pass = true;
    if (a.check) {
        std::vector<float> ref(n);
        conv2d_seq(in.data(), ref.data(), k.data(), a.H, a.W, a.K);
        err = compare(ref, out, sum_abs_terms(in, k, a.H, a.W, a.K), a.K);
        pass = within_tol(err);
    }

    if (!a.dump_dir.empty()) {
        std::string base = a.dump_dir + "/" + std::to_string(a.H) + "x" +
                           std::to_string(a.W) + "_k" + std::to_string(a.K) + "_";
        if (!write_bin(base + "in.bin", in) || !write_bin(base + "k.bin", k) ||
            !write_bin(base + a.impl + "_out.bin", out)) {
            std::fprintf(stderr, "failed to write dump files to %s\n", a.dump_dir.c_str());
            return 1;
        }
    }

    const std::string bx = cuda ? std::to_string(a.bx) : "";
    const std::string by = cuda ? std::to_string(a.by) : "";
    if (a.header)
        std::printf("impl,H,W,K,bx,by,iters,median_ms,min_ms,"
                    "h2d_ms,kernel_ms,d2h_ms,alloc_ms,"
                    "max_abs_err,max_rel_err,err_ratio,nonfinite,pass\n");
    std::printf("%s,%d,%d,%d,%s,%s,%d,%.4f,%.4f,%s,%s,%s,%s,%.3e,%.3e,%.3e,%zu,%d\n",
                a.impl.c_str(), a.H, a.W, a.K, bx.c_str(), by.c_str(), a.iters,
                median(t.call_ms),
                *std::min_element(t.call_ms.begin(), t.call_ms.end()),
                field(t.h2d_ms).c_str(), field(t.kernel_ms).c_str(),
                field(t.d2h_ms).c_str(), field(t.alloc_ms).c_str(),
                err.max_abs, err.max_rel, err.max_ratio, err.nonfinite,
                pass ? 1 : 0);
    return pass ? 0 : 2;
}
