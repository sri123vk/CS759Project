# Performance Comparison of Sequential, OpenMP, and CUDA-Based 2D Convolution

ECE 759 · Fall 2026 · Final Project

**Team:** Srimathi Ravisankar, Kshitija Sunil Rangrej, Vanessa Bennett, Frey Wysong

## Overview

One 2D convolution, implemented three ways (sequential C++, OpenMP, CUDA)
and benchmarked across image sizes 256²–2048² and filter sizes 3×3, 5×5, 7×7.
We also study OpenMP thread count and scheduling, and CUDA thread-block shape.

## Specification

Every implementation computes, with `R = K / 2`:

```
out[r][c] = Σᵢ Σⱼ k[i][j] · in[r+i−R][c+j−R]
```

| | |
|---|---|
| Operation | Cross-correlation (filter not flipped), as in image-processing and CNN libraries |
| Precision | FP32 input, filter, output, and accumulator |
| Layout | Row-major; pixel `(r, c)` at `in[r*W + c]` |
| Filter | Square, odd `K` ∈ {3, 5, 7}, centered |
| Boundary | Zero padding; output is `H × W` |
| Sum order | Filter row `i`, then column `j` |

Shared interface (`include/conv2d.h`):

```cpp
void conv2d_<variant>(const float* in, float* out, const float* k,
                      int H, int W, int K);
```

## Methodology

**Inputs.** Uniform in [−1, 1] from fixed seeds. Exportable as raw float32 for
Python cross-checks.

**Correctness.** Every implementation is checked against hand-derived answers
and an independent double-precision loop (`make test`), and its output at
benchmark sizes is checked against
`scipy.signal.correlate2d(in, k, mode="same", boundary="fill")` (`make validate`).
See [Tests](#tests). Pass: no NaN/Inf in the output, and at every pixel
`|got − ref| ≤ 2·γ(K²)·Σ|k·in|` where `γ(n) = n·u / (1 − n·u)`, `u = 2⁻²⁴`.
This is the standard FP32 dot-product error bound (Higham §3.1). It holds
for any summation order, with or without FMA, so GPU results need not match
bit-for-bit. The CSV reports `err_ratio` (worst error ÷ bound, pass ≤ 1);
max absolute and relative error are reported for information only.

**Timing.** Warm-up runs, then the median of N timed runs.

- CPU: `std::chrono::steady_clock`
- CUDA: the context is created and device memory allocated before timing
  (allocation reported separately as `alloc_ms`). End-to-end time
  (`median_ms`) is host wall-clock for host→device copy + kernel +
  device→host copy. `h2d_ms`, `kernel_ms`, `d2h_ms` are per-stage
  `cudaEvent` times (medians).

All reported numbers come from one cluster node. Laptop timings are for
development only.

## Layout

```
include/      interface, timing, and I/O helpers
src/          implementations and benchmark driver
tests/        C++ correctness tests (make test)
validation/   Python check against SciPy / PyTorch (make validate)
scripts/      benchmark sweeps and Slurm jobs
results/      raw CSV output and hardware info
plots/        figures
report/       final report
```

## Build and run

```
make                                         # builds bin/conv
./bin/conv --impl seq --size 1024 --K 5 --header
./bin/conv --impl cuda --size 2048 --K 7 --block 16 16 --check
```

CUDA is built in automatically when `nvcc` is on the PATH (`make CUDA=0`
to skip it). To target a specific GPU, e.g. `make NVCCFLAGS="-O3 -std=c++17 -arch=sm_75"`.

Options: `--size N` (or `--H`, `--W`), `--K`, `--iters`, `--warmup`,
`--block BX BY` (CUDA thread block, default 32 8), `--check`, `--dump DIR`,
`--header`. Output is one CSV row per run with columns
`impl,H,W,K,bx,by,iters,median_ms,min_ms,h2d_ms,kernel_ms,d2h_ms,alloc_ms,max_abs_err,max_rel_err,err_ratio,nonfinite,pass`;
CUDA-only columns are empty for CPU implementations.

## Tests

`make test` runs the unit tests and then the external check against SciPy.
Run it after every change and before every benchmark sweep. Nothing runs it
automatically (there is no CI).

```
make test                                  # unit tests, then the external check if SciPy is installed
make test VALIDATE=1                       # same, but a missing SciPy is an error (use before benchmarking)
make test VALIDATE=0                       # unit tests only (fast, for development)
make validate                              # external check only
./bin/test_conv --list                     # implementations compiled into the unit tests
./bin/test_conv --impl cuda -v             # only names containing "cuda"; also print passes
./bin/conv --impl cuda --size 2048 --K 7 --check   # one run, compared with seq
```

If Python with numpy and scipy (or numpy and torch) isn't available,
`make test` still passes on the unit tests alone and prints
`*** SKIPPED: external check ...`. Don't treat that as a full pass.
`VALIDATE=1` turns the skip into a failure.

There are three kinds of check:

| Check | Compared against | Sizes | Command |
|---|---|---|---|
| Unit tests, `tests/test_conv.cpp` | Hand-derived answers, and a double-precision loop written separately in the test file | 1×1 to 1000×1000 | `make test` |
| External check, `validation/validate.py` | SciPy `correlate2d` (or PyTorch `conv2d`) in float64 | 2048² and 1531×2777 by default | `make test`, or on its own `make validate` |
| Driver `--check` | `conv2d_seq` | Any | `bin/conv ... --check` |

### Unit tests (`make test`)

Every test runs on every implementation that is compiled in:

| Name | Built in when | Requirement |
|---|---|---|
| `seq` | Always | Within the error bound |
| `omp/<static\|dynamic>/t<1,2,3,max>` | `src/conv2d_omp.cpp` exists | Bit-identical to `seq`, since each pixel is summed in the same order |
| `cuda/32x8` (the one-call `conv2d_cuda`), `cuda/8x8`, `16x16`, `32x32`, `128x1`, `1x256`, `7x3` | `nvcc` is on the PATH (`make CUDA=0` to skip) | Within the error bound (FMA changes the rounding). `7x3` leaves partial blocks along every edge |

Every call also checks for memory errors. The output buffer starts as NaN,
so pixels that are never written fail. Guard values on both sides of the
buffer catch out-of-bounds writes. The input and filter must be unchanged
afterwards.

| Group | Input | Expected output | Catches |
|---|---|---|---|
| identity | Center tap = 1, K = 1–7 | The input | Basic indexing |
| shift | One tap = 1 at every filter position, non-square images | The input shifted, zeros filled in | Swapped rows/columns, off-by-one padding |
| impulse | One pixel = 1, at the center and each corner | The filter rotated 180° | Convolution instead of cross-correlation, edge clipping |
| ones | All-ones image and filter, including images smaller than the filter | Count of in-bounds taps | Zero-padding errors |
| random | Uniform [−1, 1], 12 shapes (1×1, 1×17, 17×1, primes, 257×263, 1000×1000, …) × K = 1, 3, 5, 7 | Within the error bound of the double-precision loop | General arithmetic and indexing |
| repeatability | 513×517, K = 7, run 3 times | Identical bits every run | Data races, reads of uninitialized memory |

The hand-derived cases use only small integers and products with 0 or 1, so
any implementation must reproduce them exactly whatever the summation order.
The random cases use the error bound from [Methodology](#methodology).

Exit status: 0 if every check passes, 1 if any fails, 2 for bad arguments.
At most 5 failures are printed per implementation per group, and the summary
table gives the totals.

**Adding an implementation:** add an entry to `all_impls()` in
`tests/test_conv.cpp` with a name, the function, an optional setup step run
before each call, and whether the output must match `seq` bit for bit. Set
that flag only if the implementation sums each pixel in `seq`'s order.

### External check (part of `make test`; on its own, `make validate`)

1. Runs `bin/conv --dump` once for each shape and implementation, writing
   raw float32 files to `data/validate/` (gitignored, about 200 MB with the
   defaults). Old `.bin` files there are deleted first.
2. Runs `validation/validate.py data/validate`. It first checks the reference
   library itself against known answers: single taps at every filter position,
   and a random filter checked against a direct loop. A library that flips the
   filter or centers its output differently stops the run there. Then every
   output is compared with the library's float64 result using the same error
   bound as the unit tests.

Requirements: Python 3 with numpy and scipy. Without scipy it uses PyTorch
(`--ref torch`). Settings, all optional:

```
# VALIDATE_SHAPES: HxWxK; VALIDATE_IMPLS default: seq, plus cuda if built in
make validate PYTHON=python3 \
              VALIDATE_SHAPES="2048x2048x3 1531x2777x7" \
              VALIDATE_IMPLS="seq cuda" \
              VALIDATE_DIR=data/validate
```

To check dumps you made yourself:
`python3 validation/validate.py DIR [--ref auto|scipy|torch] [-v]`. Exit
status: 0 if all pass, 1 if any fail, 2 for missing or malformed files.

### Not hooked up yet

- **OpenMP.** Being written separately. The build and the unit tests switch
  it on automatically once `src/conv2d_omp.cpp` exists and defines
  `conv2d_omp`: the Makefile checks for that exact path and the tests call
  that exact name. The tests set the schedule with `omp_set_schedule`, so the
  `static` and `dynamic` variants only differ if the loop uses
  `schedule(runtime)`. For the external check, run
  `make validate VALIDATE_IMPLS="seq omp cuda"`.
- **`src/conv2d_cuda_opt.cu`.** Not built or tested. It defines `conv2d_cuda`
  and `conv2d_kernel`, the same names as `src/conv2d_cuda.cu`, so one of them
  has to be renamed before both can be linked. Then add it to the Makefile and
  to `all_impls()`.
- **`convolve()` in `src/conv2d_seq_opt_for_largesize.cpp`.** Not built or
  tested. Its signature differs (`size_t` dimensions, filter after the
  dimensions), so `all_impls()` needs a short wrapper for it.
- **`--check` compares against `conv2d_seq`**, not an independent reference.
  A bug shared by `seq` and another implementation would pass it. `make validate`
  covers this, but only for the shapes it runs.
- **`make validate` covers CUDA only with the default 32×8 block.** Dump file
  names don't include the block shape. Other block shapes are covered by
  `make test`, up to 1000×1000.
- **No tests for invalid input** (even or negative K, zero sizes, null
  pointers). `bin/conv` rejects a bad K itself, but the implementations are
  not tested with one.
- **No CI.** `make test` only runs when someone runs it, and with the default
  `VALIDATE=auto` it skips the external check on machines without SciPy.

## Status

- [x] Sequential baseline
- [ ] OpenMP
- [ ] CUDA 
- [ ] Benchmarks and plots
- [ ] Report
