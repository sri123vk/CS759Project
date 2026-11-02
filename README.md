# Performance Comparison of Sequential, OpenMP, and CUDA-Based 2D Convolution

ECE 759, Fall 2026 — Final Project

Team: Srimathi Ravisankar, Kshitija Sunil Rangrej, Vanessa Bennett, Frey Wysong

## Overview

We implement the same 2D convolution three ways — a sequential C++ baseline,
an OpenMP multicore version, and a CUDA GPU version — and compare their
performance across image sizes (256² to 2048²) and filter sizes (3×3, 5×5, 7×7).
We also study OpenMP thread count / scheduling and CUDA thread-block shape.

## Operation specification

All implementations must compute exactly this operation so results are
directly comparable.

| Item | Decision |
|---|---|
| Operation | 2D cross-correlation (the filter is **not** flipped). This is what image-processing and CNN libraries call "convolution". |
| Data type | FP32 (`float`) for image, filter, and output |
| Layout | Row-major, contiguous. Pixel `(r, c)` is at `in[r * W + c]` |
| Filter size | Square, odd `K` (3, 5, 7). Center at `(K/2, K/2)` |
| Boundary | Zero padding; output has the same size as the input (`H × W`) |
| Accumulation | Single-precision accumulator, loop order `i` (filter row) then `j` (filter column) |

Formally, with `R = K / 2`:

```
out[r][c] = sum_{i=0}^{K-1} sum_{j=0}^{K-1} k[i][j] * in[r + i - R][c + j - R]
```

where any `in[y][x]` with `y` or `x` outside the image is treated as 0.

### Common interface

Every implementation exposes the same signature (see `include/conv2d.h`):

```cpp
void conv2d_<variant>(const float* in, float* out, const float* k,
                      int H, int W, int K);
```

### Inputs

Image and filter values are generated from a fixed seed (uniform in [-1, 1])
so runs are reproducible. Inputs can also be written to / read from raw
binary files for cross-checking with Python.

### Correctness

Each parallel output is compared against the sequential output, and the
sequential output is compared against a NumPy/SciPy reference
(`scipy.signal.correlate2d(in, k, mode="same", boundary="fill")`).
A result passes if the maximum absolute error is at most `1e-4` and the
maximum relative error (on entries with magnitude > 1e-3) is at most `1e-5`.
GPU results are not expected to match bit-for-bit because of FMA contraction
and different operation order.

### Timing methodology

- Wall-clock CPU timing uses `std::chrono::steady_clock`.
- CUDA kernel time uses `cudaEvent` pairs; end-to-end CUDA time also includes
  host-to-device and device-to-host copies.
- Each configuration runs a few warm-up iterations, then reports the median
  of N timed iterations.
- All reported numbers come from the same cluster node. Laptop timings are
  only for development.

## Repository layout

```
include/      shared headers (interface, timing, I/O helpers)
src/          implementations and the benchmark driver
validation/   Python reference and correctness checks
scripts/      benchmark sweeps and Slurm job scripts
results/      raw benchmark output (CSV) and hardware info
plots/        generated figures
report/       final report
```

## Building

```
make            # sequential + OpenMP driver (bin/conv)
make cuda       # CUDA driver (requires nvcc)
```
