
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
//   cuda_opt/<bx>x<by>                    when built with -DCONV_HAVE_CUDA_OPT
//   cuda_const, cuda_tiled_const, cuda_multiout /<bx>x<by>
//                                         when built with -DCONV_HAVE_CUDA_VARIANTS
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

#if defined(CONV_HAVE_OMP) || defined(CONV_HAVE_OMP_SIMD) || defined(CONV_HAVE_OMP_BLOCKED)
#define CONV_HAVE_ANY_OMP
#include <omp.h>
#endif

namespace {

using ConvFn = std::function<void(const float*, float*, const float*, int, int, int)>;
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
    const struct { const char* name; ConvFn fn; bool bitwise; } omps[] = {
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




