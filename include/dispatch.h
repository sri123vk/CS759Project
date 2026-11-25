// Dispatcher: picks the implementation predicted to be fastest for a given
// problem (proposal stretch goal: choose sequential, OpenMP or CUDA by size).
//
// The prediction uses a cost model fitted to our own Euler measurements by
// scripts/fit_dispatch.py (coefficients in include/dispatch_model.h). Only
// implementations that are compiled in, and that accept the problem, are
// considered: the constant-memory CUDA kernels need C*K*K <= 4096.
#pragma once

#include <string>

struct DispatchChoice {
    std::string impl;     // name as accepted by bin/conv --impl
    bool gpu = false;     // true for the CUDA implementations
    double predicted_ms;  // model's estimate for one call (copies included for CUDA)
};

// The choice for C input channels of H x W with a K x K filter. The
// environment variable CONV_AUTO_DEVICE=cpu or =gpu restricts the choice to
// that side (for tests and experiments); unset, both are considered.
DispatchChoice conv2d_choose(int C, int H, int W, int K);

// Same interface as conv2d_seq: runs the implementation conv2d_choose picks.
// CUDA choices allocate device memory, copy, run with a 32x8 block and free,
// like conv2d_cuda.
void conv2d_auto(const float* in, float* out, const float* k,
                 int C, int H, int W, int K);
