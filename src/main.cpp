// Benchmark driver for the CPU implementations.
//
// Usage:
//   conv --impl seq|omp --size N [--K 3] [--iters 10] [--warmup 2]
//        [--check] [--dump DIR] [--header]
//
// Prints one CSV row per run (use --header to print the column names first).
// --check  also runs the sequential version and compares against it
// --dump   writes in/k/out as raw float32 files for the Python validator

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
    bool check = false, header = false;
    std::string dump_dir;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --impl seq|omp --size N [--H h --W w] [--K k]\n"
        "          [--iters n] [--warmup n] [--check] [--dump dir] [--header]\n",
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
        else if (is("--check")) a.check = true;
        else if (is("--header")) a.header = true;
        else if (is("--dump")) a.dump_dir = next();
        else usage(argv[0]);
    }
    if (a.K <= 0 || a.K % 2 == 0) {
        std::fprintf(stderr, "K must be a positive odd number\n");
        std::exit(1);
    }
    if (a.H <= 0 || a.W <= 0 || a.iters <= 0 || a.warmup < 0) usage(argv[0]);
    return a;
}

using ConvFn = void (*)(const float*, float*, const float*, int, int, int);

ConvFn pick(const std::string& impl) {
    if (impl == "seq") return conv2d_seq;
    std::fprintf(stderr, "unknown impl: %s\n", impl.c_str());
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    ConvFn fn = pick(a.impl);

    const size_t n = (size_t)a.H * a.W;
    // out starts as NaN so any pixel an implementation never writes fails --check.
    std::vector<float> in(n), k((size_t)a.K * a.K),
        out(n, std::numeric_limits<float>::quiet_NaN());
    fill_random(in, 759);
    fill_random(k, 42);

    for (int i = 0; i < a.warmup; ++i) fn(in.data(), out.data(), k.data(), a.H, a.W, a.K);

    std::vector<double> times;
    times.reserve(a.iters);
    for (int i = 0; i < a.iters; ++i) {
        times.push_back(time_ms([&] { fn(in.data(), out.data(), k.data(), a.H, a.W, a.K); }));
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

    if (a.header)
        std::printf("impl,H,W,K,iters,median_ms,min_ms,"
                    "max_abs_err,max_rel_err,err_ratio,nonfinite,pass\n");
    std::printf("%s,%d,%d,%d,%d,%.4f,%.4f,%.3e,%.3e,%.3e,%zu,%d\n",
                a.impl.c_str(), a.H, a.W, a.K, a.iters, median(times),
                *std::min_element(times.begin(), times.end()),
                err.max_abs, err.max_rel, err.max_ratio, err.nonfinite,
                pass ? 1 : 0);
    return pass ? 0 : 2;
}
