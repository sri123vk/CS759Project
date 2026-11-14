# Benchmark

Scripts that run the experiments in the proposal and produce the figures and
tables for the final report. They drive the existing benchmark driver
(`bin/conv`) and don't change any implementation code.

| Script | What it does |
|---|---|
| `scripts/bench.sh` | runs the experiment sweeps; one CSV per experiment |
| `scripts/bench_torch.py` | PyTorch / cuDNN reference on the same problems, same CSV format |
| `scripts/plot.py` | figures (PNG) and `summary.md` tables for the report |
| `scripts/sysinfo.sh` | hardware, compiler and git info, saved with every run |
| `scripts/euler.slurm` | build, test, benchmark and plot on one cluster node |

## Quick start

```
make && make test                        # driver + correctness tests must pass first
scripts/bench.sh --quick                 # ~10 s smoke run of every experiment
scripts/bench.sh -o results/final all    # full sweep -> results/final/*.csv
python3 scripts/plot.py results/final    # -> plots/final/*.png + summary.md
sbatch scripts/euler.slurm               # all of the above on a cluster node
```

The report's numbers should come from one cluster run, as the README says.
Laptop runs are for checking the scripts.

## Experiments

| Experiment | Sweep (defaults) | Proposal item |
|---|---|---|
| `baseline` | each implementation × 256², 512², 1024², 2048² × K = 3, 5, 7. OpenMP: all cores, static. CUDA: 32×8 block | execution time; OpenMP/CUDA speedup vs sequential; differences across image and filter sizes; CUDA kernel-only vs end-to-end (the CUDA rows carry `h2d_ms`, `kernel_ms`, `d2h_ms`) |
| `omp` | OpenMP implementations × threads 1, 2, 4, …, nproc × `static`/`dynamic`/`guided`, at 512² and 2048², K = 3, 7 | OpenMP thread scaling; static vs dynamic scheduling |
| `cuda_blocks` | CUDA implementations × 13 block shapes (32×8, 16×16, 128×1, 1×128, …), 1024² and 2048², K = 3, 7 | CUDA thread-block configuration; global-memory access patterns (1×N blocks are uncoalesced) |
| `torch` | PyTorch `conv2d`, cuDNN on the GPU and PyTorch's CPU convolution (oneDNN), same problems as `baseline` | review: "compare against cuDNN or PyTorch's conv2d" |

All sweep parameters can be overridden with environment variables, listed at
the top of `scripts/bench.sh`. For example, to extend the sweep toward the
reviewers' "deeper studies":
`SIZES="256 512 1024 2048 4096" KS="3 5 7 9 11" scripts/bench.sh baseline`.

### Hypotheses → evidence

| Hypothesis | Where to look |
|---|---|
| H1: speedup grows with workload | `fig2_speedup.png`; "Speedup over seq" table |
| H2: for small workloads, overhead outweighs the benefit | `fig2_speedup.png` (points below the 1× line); "Smallest image size where the implementation beats seq"; `fig8_fastest.png` |
| H3: CUDA depends on block configuration and transfers | `fig6_cuda_blocks.png`; `fig3_cuda_kernel_vs_e2e.png`; `fig4_cuda_breakdown.png`; "Transfer share" table |
| Stretch goal: dispatcher thresholds | `fig8_fastest.png` and the "Fastest implementation per workload" table give the size/filter regions where seq, OpenMP or CUDA wins |

### Report outputs (`plot.py`)

| File | Shows |
|---|---|
| `fig1_time.png` | time per call vs image size, one panel per filter size |
| `fig2_speedup.png` | end-to-end speedup over seq, with the 1× line |
| `fig3_cuda_kernel_vs_e2e.png` | CUDA speedup counting the kernel only (solid) vs including copies (dotted) |
| `fig4_cuda_breakdown.png` | share of CUDA time in host→device copy, kernel, device→host copy |
| `fig5_omp_scaling.png` | speedup vs threads for each schedule, with the ideal line |
| `fig6_cuda_blocks.png` | kernel time for each block shape, relative to the best shape |
| `fig7_throughput.png` | GFLOP/s (kernel only for GPU rows), including the cuDNN / PyTorch reference |
| `fig8_fastest.png` | the fastest of our implementations for each workload |
| `summary.md` | tables: times, speedups, crossover sizes, fastest implementation, OpenMP speedup and efficiency, block-shape times, our kernels vs cuDNN, any failed correctness checks |

Each implementation keeps the same color and marker in every figure.

## Adding implementations

The scripts benchmark every name in `IMPLS` that `bin/conv --impl NAME` accepts.
The default list already includes the names we'd likely use for the
reviewers' suggested optimizations (`omp_simd`, `omp_blocked`, `cuda_const`,
`cuda_tiled`, `cuda_tiled_mo`). A name the driver doesn't know yet is just
skipped. Once a teammate adds a variant to the driver, it shows up in the next
run with no script changes. For a different name, add it to the default
`IMPLS` in `bench.sh` or set it for one run:

```
IMPLS="seq omp cuda cuda_shared" scripts/bench.sh baseline cuda_blocks
```

Naming convention used by the sweeps:
- Names starting with `omp` run in the thread/schedule sweep. They get
  `OMP_NUM_THREADS` and `OMP_SCHEDULE` from the environment, so the loop must
  use `schedule(runtime)`, as `conv2d.h` specifies for `conv2d_omp`.
- Names starting with `cuda` run in the block-shape sweep and get `--block BX BY`.
- `seq` is the baseline for every speedup.

## Methodology

- **Correctness.** Every run except `seq` uses `--check`, which compares against
  `seq` with the FP32 error bound described in the README (`CHECK=0` turns it off).
  A failing row is still recorded. It's logged in `errors.log` and listed in
  `summary.md`. `euler.slurm` runs `make test` first and stops if it fails.
- **Repetitions.** 2 warm-up runs, then 50 timed runs up to 512², 20 at 1024²,
  and 10 above. Short runs get more samples because they're noisier. We
  report the median; `ITERS` and `WARMUP` override these counts.
- **CPU.** OpenMP threads are pinned (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`)
  so the scaling curves are stable. The schedule is always set explicitly,
  because without `OMP_SCHEDULE`, `schedule(runtime)` falls back to
  libgomp's default.
- **CUDA.** Timing follows the driver: the context is created and device
  memory allocated before timing. `median_ms` is end-to-end (copy in, kernel,
  copy out); stage times come from cudaEvents.
- **cuDNN reference.**
  - TF32 is turned off, so the arithmetic is pure FP32 like ours.
  - `cudnn.benchmark` is on, so cuDNN picks its fastest algorithm during the warm-up runs.
  - Stages are timed the same way as in the driver.
  - Its `median_ms` includes some Python overhead, so compare small problems on `kernel_ms`.
  - cuDNN may use Winograd or FFT algorithms. An `err_ratio` above 1 for
    `torch_cudnn` can mean a different algorithm, not a wrong answer.
- **Reproducibility.** `sysinfo.txt` in each results directory records the CPU,
  GPU, compiler, Makefile flags, git commit and Slurm job.

### CSV format

Each file has `threads,schedule` followed by `bin/conv`'s own columns,
taking the header from `bin/conv --header`. If the driver gains columns
later, the CSVs follow automatically. `threads` is 1 for seq and empty for
CUDA; `schedule` is set only for OpenMP rows. The FLOP count
(2·K²·H·W) and GFLOP/s are computed in `plot.py`.

## Not covered yet

These need support in the driver first. Once it's there, they're one more
experiment function in `bench.sh`:
- **Multi-channel input** (review). Needs `Cin`/`Cout` options in `bin/conv`.
- **Pinned vs pageable host memory.** This would show how much of the
  transfer cost (H3) can be recovered. Needs a `--pinned` option.

## Requirements

- What `make` already needs: g++ with OpenMP, and nvcc for CUDA.
- bash and coreutils.
- For plots: Python 3 with pandas and matplotlib.
- For the `torch` experiment: PyTorch with CUDA (conda or
  `pip install --user torch` on the cluster). The experiment is skipped if
  PyTorch is missing.
- Check the partition, module names and `-c` in `scripts/euler.slurm` before
  the first cluster run.
