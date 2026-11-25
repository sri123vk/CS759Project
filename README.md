# Performance Comparison of Sequential, OpenMP, and CUDA-Based 2D Convolution

ECE 759 · Fall 2026 · Final Project

**Team:** Srimathi Ravisankar, Kshitija Sunil Rangrej, Vanessa Bennett, Frey Wysong

## Overview

One 2D convolution, implemented three ways (sequential C++, OpenMP, CUDA)
and benchmarked across image sizes 256²–2048² and filter sizes 3×3, 5×5, 7×7.
We also study OpenMP thread count and scheduling, CUDA thread-block shape,
shared-memory tiling, constant-memory filters, and multiple outputs per thread.

## Specification

Every implementation computes, with `R = K / 2`:

```
out[r][c] = Σ_ch Σᵢ Σⱼ k[ch][i][j] · in[ch][r+i−R][c+j−R]
```

| | |
|---|---|
| Operation | Cross-correlation (filter not flipped), as in image-processing and CNN libraries |
| Precision | FP32 input, filter, output, and accumulator |
| Channels | `C` input channels, summed into one output channel. Every implementation (`seq`, OpenMP and CUDA) supports `C > 1` |
| Layout | Row-major, channel-major (CHW); pixel `(ch, r, c)` at `in[(ch*H + r)*W + c]`, filter tap `(ch, i, j)` at `k[(ch*K + i)*K + j]` |
| Filter | Square, positive odd `K`, centered, one `K × K` slice per input channel; benchmarks commonly use 3, 5, and 7 |
| Boundary | Zero padding; output is `H × W` |
| Sum order | Channel `ch`, then filter row `i`, then column `j` |

Interfaces (`include/conv2d.h`). The sequential, OpenMP and direct CUDA
versions all take the channel count:

```cpp
void conv2d_seq(const float* in, float* out, const float* k,
                int C, int H, int W, int K);   // also conv2d_cuda
void conv2d_<omp variant>(const float* in, float* out, const float* k,
                          int C, int H, int W, int K);
```

The CUDA benchmark path (`cuda_conv_alloc(C, H, W, K)`, `cuda_conv_run`,
`cuda_conv_free`) and the additional CUDA variants use the staged interface
declared in `include/conv2d.h`; the variants take C from the allocation.
Shared-tile variants load one channel's tile at a time, so their shared
memory doesn't grow with C. Constant-memory variants keep all C filter slices
in a 4096-float symbol, so they need `C·K² ≤ 4096` (odd `K ≤ 63` for one
channel, up to C = 83 for K = 7) and refuse larger filters.

## Methodology

**Inputs.** Uniform in [−1, 1] from fixed seeds. Exportable as raw float32 for
Python cross-checks.

**Correctness.** The compiled implementations are checked against hand-derived
answers and an independent double-precision loop (`make test`). The separate
`make validate` target checks selected shapes and implementations against
`scipy.signal.correlate2d(in, k, mode="same", boundary="fill")` or PyTorch.
By default, external validation selects `seq` and, when built, `cuda`; other
variants can be included with `VALIDATE_IMPLS`.
See [Tests](#tests). Pass: no NaN/Inf in the output, and at every pixel
`|got − ref| ≤ 2·γ(C·K²)·Σ|k·in|` where `γ(n) = n·u / (1 − n·u)`, `u = 2⁻²⁴`.
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

For comparable results, run experiments on the same cluster node type and
record the hardware and software configuration. Laptop timings are useful
for development only.

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
make -B CUDA=1 bin/conv bin/test_conv        # force a CUDA + OpenMP rebuild
./bin/conv --impl seq --size 1024 --K 5 --header
./bin/conv --impl seq --size 1024 --K 5 --C 3 --check    # 3 input channels
./bin/conv --impl cuda --size 2048 --K 7 --block 16 16 --check
./bin/conv --impl cuda_opt --size 2048 --K 7 --block 32 8 --check
```

CUDA is built in automatically when `nvcc` is on the PATH (`make CUDA=0`
to skip it). To target a specific GPU, e.g. `make NVCCFLAGS="-O3 -std=c++17 -arch=sm_75"`.
Use `make -B` after changing build options because Make may not detect changes
in flags. On Euler, load an appropriate CUDA module before building and run
CUDA tests and benchmarks on a GPU compute node.

Options: `--size N` (or `--H`, `--W`), `--K`, `--C` (input channels,
default 1; values above 1 are accepted by every implementation), `--iters`, `--warmup`,
`--block BX BY` (CUDA thread block, default 32 8), `--block auto`, `--check`, `--dump DIR`,
`--header`. Output is one CSV row per run with columns
`impl,H,W,K,C,bx,by,iters,median_ms,min_ms,h2d_ms,kernel_ms,d2h_ms,alloc_ms,max_abs_err,max_rel_err,err_ratio,nonfinite,pass`;
CUDA-only columns are empty for CPU implementations.
`--block auto` tries supported candidate shapes on the current workload and
reports the selected shape and tuning time separately. Tuning is excluded from
the reported benchmark times. `--check` must be specified for the `pass` field
to represent a correctness comparison.

CUDA implementations available through `--impl`:

| Name | Implementation |
|---|---|
| `cuda` | Direct baseline; one output per thread. |
| `cuda_opt` | Shared-memory input tiles with halos and a global-memory filter. Fails with an error if the tile does not fit in shared memory. |
| `cuda_const` | Direct kernel with filter coefficients in constant memory. |
| `cuda_tiled_const` | Shared-memory input tiles and a constant-memory filter. |
| `cuda_multiout` | Two outputs per thread with a global-memory filter. |

The OpenMP versions are `omp`, `omp_simd`, `omp_blocked` and `omp_tiled`
(tiles of 4 output rows × 16 columns whose running sums stay in registers
across all channels and filter taps).

### Dispatcher (`--impl auto`)

`--impl auto` runs the implementation predicted to be fastest for the
problem (C, H, W, K) among those compiled in; the choice is printed on
stderr, and the CSV row is that implementation's. The prediction is a cost
model per implementation,

```
t_ms = c0 + c1*H*W*C + c2*H*W + c3*H*W*C*K^2
```

(fixed cost per call, per input element, per output element, per
multiply-add), fitted to an Euler run by `scripts/fit_dispatch.py`, which
writes the coefficients to `include/dispatch_model.h` and a report of how
often the model picks the fastest implementation. Refit after a new run:

```
python3 scripts/fit_dispatch.py results/<run> --report results/<run>/dispatch.md
```

The model is for the node type it was measured on (16 OpenMP threads, one
RTX 4000 Ada). `CONV_AUTO_DEVICE=cpu` or `=gpu` restricts the choice to one
side. From C++, `conv2d_auto()` in `include/dispatch.h` has the same
interface as `conv2d_seq`.

## Tests

`make test` runs the unit tests and then the external check if NumPy and SciPy
or NumPy and PyTorch are available.
Run it after every change and before every benchmark sweep. There is no CI;
the provided Euler benchmark job also runs the tests.

```
make test                                  # unit tests, then external check if dependencies are installed
make test VALIDATE=1                       # require the external check
make test VALIDATE=0                       # unit tests only (fast, for development)
make validate                              # external check only
./bin/test_conv --list                     # implementations compiled into the unit tests
./bin/test_conv --impl cuda -v             # only names containing "cuda"; also print passes
./bin/test_conv --impl cuda_opt/7x3        # test the optimized kernel with partial blocks
./bin/test_conv --impl cuda_tiled_const/7x3
./bin/conv --impl cuda --size 2048 --K 7 --check   # one run, compared with seq
```

If Python with numpy and scipy (or numpy and torch) isn't available,
`make test` still passes on the unit tests alone and prints
`*** SKIPPED: external check ...`. Don't treat that as a full pass.
`VALIDATE=1` turns the skip into a failure.
`--impl` matches a substring, so `--impl cuda` selects matching CUDA variants;
use `--list --impl cuda` to see exactly what will run.

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
| `omp`, `omp_simd`, `omp_blocked`, `omp_tiled` with `static`, `dynamic`, or `guided` and multiple thread counts | Corresponding `src/omp/conv2d_*.cpp` exists | Checked against the sequential and independent references |
| `cuda/32x8` (the one-call `conv2d_cuda`), `cuda/8x8`, `16x16`, `32x32`, `64x4`, `64x16`, `128x1`, `256x4`, `1024x1`, `1x256`, `4x256`, `7x3`, `33x5` | `nvcc` is on the PATH (`make CUDA=0` to skip) | Within the error bound (FMA changes the rounding). Includes blocks wider than a warp (up to 1024×1) and taller than wide (1×256, 4×256); `7x3` and `33x5` leave partial blocks along every edge |
| `auto`, `auto/cpu` and, with CUDA, `auto/gpu` (the dispatcher, free to pick, or restricted to one side) | Always | Whatever it picks must pass the same checks |
| All seven variants (`cuda_opt`, `cuda_const`, `cuda_tiled_const`, `cuda_multiout`, `cuda_multiout_const`, `cuda_multiout_tiled`, `cuda_multiout_tiled_const`) with `32x8`, `64x4`, `128x8`, `256x1`, `1x128`, `7x3`, `33x5` | CUDA variants are compiled in | Within the error bound. The wide blocks also give the shared-tile variants wide tiles; `7x3` and `33x5` exercise partial blocks |

The tests also check for some memory errors. The output buffer starts as NaN,
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
| multi-channel exact (`seq` and all CUDA) | C = 2, 3, 5: a filter that is nonzero in one slice only; an impulse in one channel only; all ones | That input channel; that filter slice rotated 180°; C × the in-bounds tap count | Wrong input-channel offset, input channel paired with the wrong filter slice, channels skipped or repeated |
| multi-channel random (`seq` and all CUDA) | Uniform [−1, 1], 6 shapes × C = 2, 3, 16 × K = 1, 3, 5, 7 | Within the error bound (with C·K² terms) of the double-precision loop | General multi-channel indexing |
| multi-channel configurations (`seq` and all CUDA) | 1×1, 17×1, 33×65, 128×128 × (C, K) = (4, 9), (8, 11), (32, 11), (64, 7), (83, 7), (128, 5) | Within the error bound of the double-precision loop | Many channels, large filters, and C·K² = 4067, just under the constant-memory limit of 4096 |
| multi-channel repeatability (`seq` and all CUDA) | 257×263, C = 8, K = 5, run 3 times | Identical bits every run | A missing barrier between channels in the shared-tile kernels |
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
The function takes the channel count, `(in, out, k, C, H, W, K)`. Set the
last field, `multichannel`, to `true` so that the multi-channel groups also
run on it (every current implementation does).

### External check (part of `make test`; on its own, `make validate`)

1. Runs `bin/conv --dump` once for each shape and implementation, writing
   raw float32 files to `data/validate/` (gitignored, about 300 MB with the
   defaults). Old `.bin` files there are deleted first.
2. Runs `validation/validate.py data/validate`. It first checks the reference
   library itself against known answers: single taps at every filter position,
   and a random filter checked against a direct loop. A library that flips the
   filter or centers its output differently stops the run there. Then every
   output is compared with the library's float64 result using the same error
   bound as the unit tests.

Requirements: Python 3 with NumPy and SciPy, or NumPy and PyTorch. Without SciPy it uses PyTorch
(`--ref torch`). Settings, all optional:

```
# VALIDATE_SHAPES: HxWxK, or HxWxKxC for C input channels
# VALIDATE_IMPLS default: seq and every OpenMP version built, plus cuda and
# cuda_multiout_tiled_const if built in; every implementation runs every shape
make validate PYTHON=python3 \
              VALIDATE_SHAPES="2048x2048x3 1531x2777x7 1024x768x5x3" \
              VALIDATE_IMPLS="seq cuda" \
              VALIDATE_DIR=data/validate
```

The defaults include small images (`16x16x3`, `31x17x5x4`) and four more
multi-channel shapes: `1024x768x5x3`, `257x263x7x16`, `300x200x11x8` and
`128x128x7x64`. Multi-channel dump files are prefixed `{H}x{W}_k{K}_c{C}_`.
Single-channel dumps keep the `{H}x{W}_k{K}_` prefix.

To validate all five CUDA variants, set
`VALIDATE_IMPLS="seq cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout"`
on a GPU compute node. Use a dedicated `VALIDATE_DIR`: the target deletes
existing `.bin` files in that directory before writing new dumps.

To check dumps you made yourself:
`python3 validation/validate.py DIR [--ref auto|scipy|torch] [-v]`. Exit
status: 0 if all pass, 1 if any fail, 2 for missing or malformed files.

### Not hooked up yet

- **Multiple output channels.** Every implementation takes multi-channel
  input, but multiple output channels (`Cout`, one filter per output) are
  not implemented anywhere.
- **`convolve()` in `src/seq/conv2d_seq_opt_for_largesize.cpp`.** Not built or
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
  `VALIDATE=auto` it skips the external check if neither SciPy nor PyTorch
  is available alongside NumPy.

## Benchmarks

The scripts in `scripts/` run the experiments for the report by calling
`bin/conv` across a set of configurations. They don't change any
implementation code. For more detail, see [BENCHMARK.md](BENCHMARK.md).

```
make && make test                        # benchmark only code that passes its tests
scripts/bench.sh --quick                 # reduced sweep of every experiment
scripts/bench.sh -o results/final all    # full sweep -> results/final/*.csv
python3 scripts/plot.py results/final    # -> plots/final/*.png + summary.md
sbatch scripts/euler.slurm               # build, test, sweep and plot on one cluster node
sbatch scripts/cuda_variants.slurm       # test CPU/CUDA variants and run selected sweeps
```

Check the partition, resources, and time limit in each Slurm script before
submission. `cuda_variants.slurm` builds and tests the variants, then runs
`baseline`, `omp`, and `cuda_blocks`; it writes results to
`results/cuda_variants_<jobid>/`. It does not run the `torch` experiment.

What is benchmarked:

| Experiment | Configurations | Measures |
|---|---|---|
| `baseline` | every implementation × 16², 32², 64², 128², 256², 512², 1024², 2048² × K = 3, 5, 7 | time per call, speedup over `seq` (the sizes below 256² find where parallel versions stop paying off, H2), CUDA kernel-only vs end-to-end and the time spent in each stage |
| `omp` | OpenMP × 1, 2, 4, … cores threads × `static` / `dynamic` / `guided`, at 64², 512², 2048² | thread scaling and scheduling policy, including a small image |
| `cuda_blocks` | CUDA × 26 thread-block shapes: warp-wide rows, squares, rows wider than 32 (64×4 to 1024×1), columns taller than 32 (1×128 to 1×512, 4×64 to 16×64) | effect of block shape and memory coalescing |
| `torch` | PyTorch `conv2d` on the GPU when available and on the CPU, same problems as `baseline` | an external reference point; GPU execution can use cuDNN |
| `channels` | `seq` and every CUDA implementation × C = 1, 3, 8, 16, 32, 64 × 256², 512², 1024² × K = 3, 5, 7; PyTorch on the same problems | multi-channel input: time, speedup, and copy share as C grows; cuDNN on the input it's designed for |
| `channel_blocks` | CUDA × the 26 block shapes at C = 16, 1024², K = 3, 7 | block shape with multi-channel input |

Block shapes a kernel can't run (for example a shared-memory tile too large
for a 1024×1 block with a 7×7 filter) make `bin/conv` exit with status 3.
`bench.sh` records those in `skipped.log` instead of counting them as errors.

When `CHECK=1` (the default), non-sequential driver runs are checked against
`seq`, and failures are logged to `errors.log`. The script attempts to capture
hardware, compiler information, and the git commit in `sysinfo.txt`; check
that file for errors. Sweep parameters
(implementations, sizes, filter sizes, threads, block shapes, iteration
counts) can be overridden with environment variables. See
`scripts/bench.sh --help`.

The plots need Python 3 with NumPy, pandas, and matplotlib. The `torch` experiment
needs PyTorch and is skipped if it isn't installed. A CUDA-enabled PyTorch
installation and GPU are needed for the GPU library comparison.

### Plot benchmark results

After the benchmark job completes, choose its results directory under
`results/cuda_variants_<jobid>/`. It contains `baseline.csv`,
`cuda_blocks.csv`, and, when the OpenMP sweep was run, `omp.csv`.
Activate a Python environment with the plotting dependencies, then run:

```bash
cd /srv/home/rangrej/CS759Project

results="results/cuda_variants_12345"

python scripts/plot_cuda_blocks.py "$results" \
    --metric kernel_ms -o plots/cuda_comparison

python scripts/plot_cuda_blocks.py "$results" \
    --metric median_ms -o plots/cuda_comparison

python scripts/plot.py "$results" -o plots/project_report
```

Replace `/srv/home/rangrej/CS759Project` with your checkout path and `12345`
with the actual job ID. For example, job `123777` writes to
`results/cuda_variants_123777/`.

The CUDA comparison script plots the CUDA implementations in the results
and writes a PDF and comparison CSV for each metric:

- `plots/cuda_comparison/cuda_kernel_comparison.pdf` and `.csv`: GPU kernel time.
- `plots/cuda_comparison/cuda_median_comparison.pdf` and `.csv`: full-call time,
  including transfers and host overhead.

`plot.py` writes project-wide figures and `summary.md` to
`plots/project_report/`. The plots depend on the available CSVs; for example,
a cuDNN comparison is only produced if `torch.csv` is present.

`plot_cuda_blocks.py` can also read a Slurm output log for `median_ms`.
Use the benchmark CSVs for `kernel_ms`; the printed progress lines do not
contain separate kernel timings.

## Status

- [x] Sequential baseline
- [x] OpenMP
- [x] CUDA variants
- [x] Benchmark and plotting scripts
- [ ] Report
