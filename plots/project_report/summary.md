# Benchmark results

356 usable timing rows; 0 explicitly failed rows excluded from the plots.

A `pass=1` field alone does not prove `--check` was enabled; consult the benchmark command or log.

Where available, speedup vs seq uses the sequential timing from the same sweep and workload; CUDA-only sweeps fall back to the matching baseline sequential timing.

Hardware metadata is incomplete: sysinfo.txt reports a permission error.

No cuDNN benchmark rows were found; no cuDNN comparison is reported.

Nominal GFLOP/s uses 2 × H × W × K² operations; border handling can reduce actual work.


## Execution time, end-to-end (ms, median)

| workload | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const | omp | omp_blocked | omp_simd | seq |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024×1024, K=3 | 0.4248 | 0.4266 | 0.4200 | 0.4201 | 0.4271 | 0.4333 | 0.4397 | 0.5968 | 2.1508 | 0.2198 | 0.1495 | 8.5757 |
| 1024×1024, K=7 | 0.4732 | 0.4592 | 0.4743 | 0.4583 | 0.4680 | 0.4507 | 0.4833 | 0.4642 | 6.3512 | 0.8197 | 0.7162 | 31.4360 |
| 2048×2048, K=3 | 1.4597 | 1.4569 | 1.4739 | 1.4327 | 1.4666 | 1.4719 | 1.4918 | 1.4792 | 8.6048 | 0.7066 | 0.6335 | 34.2695 |
| 2048×2048, K=7 | 1.6456 | 1.6054 | 1.6129 | 1.5546 | 1.6044 | 1.5500 | 1.6834 | 1.5981 | 25.4023 | 2.8853 | 3.1520 | 125.8660 |

## Speedup over seq, end-to-end (H1, H2)

| workload | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const | omp | omp_blocked | omp_simd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024×1024, K=3 | 20.19 | 20.10 | 20.42 | 20.41 | 20.08 | 19.79 | 19.50 | 14.37 | 3.99 | 39.02 | 57.36 |
| 1024×1024, K=7 | 66.43 | 68.46 | 66.28 | 68.59 | 67.17 | 69.75 | 65.04 | 67.72 | 4.95 | 38.35 | 43.89 |
| 2048×2048, K=3 | 23.48 | 23.52 | 23.25 | 23.92 | 23.37 | 23.28 | 22.97 | 23.17 | 3.98 | 48.50 | 54.10 |
| 2048×2048, K=7 | 76.49 | 78.40 | 78.04 | 80.96 | 78.45 | 81.20 | 74.77 | 78.76 | 4.95 | 43.62 | 39.93 |

## CUDA: kernel-only speedup over seq, and share of time spent copying

| workload | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const |
|---|---|---|---|---|---|---|---|---|
| 1024×1024, K=3 | 363.38 | 381.14 | 444.34 | 492.86 | 348.61 | 381.14 | 279.34 | 298.80 |
| 1024×1024, K=7 | 416.92 | 520.46 | 472.01 | 578.93 | 451.67 | 612.79 | 365.53 | 502.98 |
| 2048×2048, K=3 | 423.60 | 449.14 | 631.11 | 689.53 | 440.48 | 484.72 | 315.27 | 341.33 |
| 2048×2048, K=7 | 458.70 | 565.18 | 569.01 | 700.81 | 536.74 | 718.82 | 416.64 | 570.30 |

Transfer share (h2d + d2h) / sum of the three stage medians; these medians are not necessarily the median full-call time:

| workload | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const |
|---|---|---|---|---|---|---|---|---|
| 1024×1024, K=3 | 94% | 95% | 95% | 96% | 94% | 95% | 93% | 95% |
| 1024×1024, K=7 | 84% | 87% | 86% | 88% | 85% | 88% | 82% | 86% |
| 2048×2048, K=3 | 94% | 95% | 96% | 97% | 95% | 95% | 93% | 93% |
| 2048×2048, K=7 | 83% | 86% | 86% | 88% | 85% | 89% | 82% | 86% |

## CUDA kernel time at the baseline block (ms)

| workload | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const |
|---|---|---|---|---|---|---|---|---|
| 1024×1024, K=3 | 0.0236 | 0.0225 | 0.0193 | 0.0174 | 0.0246 | 0.0225 | 0.0307 | 0.0287 |
| 1024×1024, K=7 | 0.0754 | 0.0604 | 0.0666 | 0.0543 | 0.0696 | 0.0513 | 0.0860 | 0.0625 |
| 2048×2048, K=3 | 0.0809 | 0.0763 | 0.0543 | 0.0497 | 0.0778 | 0.0707 | 0.1087 | 0.1004 |
| 2048×2048, K=7 | 0.2744 | 0.2227 | 0.2212 | 0.1796 | 0.2345 | 0.1751 | 0.3021 | 0.2207 |

## CUDA kernel speedup vs cuda at the same image, filter, and block (>1 is faster)

| workload | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const |
|---|---|---|---|---|---|---|---|---|
| 1024×1024, K=3 | 1.00 | 1.05 | 1.22 | 1.36 | 0.96 | 1.05 | 0.77 | 0.82 |
| 1024×1024, K=7 | 1.00 | 1.25 | 1.13 | 1.39 | 1.08 | 1.47 | 0.88 | 1.21 |
| 2048×2048, K=3 | 1.00 | 1.06 | 1.49 | 1.63 | 1.04 | 1.14 | 0.74 | 0.81 |
| 2048×2048, K=7 | 1.00 | 1.23 | 1.24 | 1.53 | 1.17 | 1.57 | 0.91 | 1.24 |

## Smallest tested image size where the implementation beats seq

| implementation | 3×3 | 7×7 |
|---|---|---|
| cuda | 1024×1024 | 1024×1024 |
| cuda_const | 1024×1024 | 1024×1024 |
| cuda_multiout | 1024×1024 | 1024×1024 |
| cuda_multiout_const | 1024×1024 | 1024×1024 |
| cuda_multiout_tiled | 1024×1024 | 1024×1024 |
| cuda_multiout_tiled_const | 1024×1024 | 1024×1024 |
| cuda_opt | 1024×1024 | 1024×1024 |
| cuda_tiled_const | 1024×1024 | 1024×1024 |
| omp | 1024×1024 | 1024×1024 |
| omp_blocked | 1024×1024 | 1024×1024 |
| omp_simd | 1024×1024 | 1024×1024 |

## Fastest tested implementation per baseline workload

| filter | 1024×1024 | 2048×2048 |
|---|---|---|
| 3×3 | omp_simd (57.4×) | omp_simd (54.1×) |
| 7×7 | cuda_multiout_tiled_const (69.7×) | cuda_multiout_tiled_const (81.2×) |

## OpenMP parallel speedup vs the same variant at one thread


### omp: 1024×1024, K=3

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.97 | 1.99 | 1.99 |
| 4.0 | 3.00 | 3.93 | 3.95 |
| 8.0 | 3.55 | 3.91 | 3.95 |

### omp: 1024×1024, K=7

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.98 | 1.99 | 1.98 |
| 4.0 | 3.58 | 3.90 | 3.90 |
| 8.0 | 4.85 | 5.10 | 4.96 |

### omp: 2048×2048, K=3

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.97 | 1.99 | 1.99 |
| 4.0 | 3.38 | 3.94 | 3.95 |
| 8.0 | 3.75 | 3.94 | 3.97 |

### omp: 2048×2048, K=7

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.97 | 1.99 | 1.99 |
| 4.0 | 3.78 | 3.91 | 3.96 |
| 8.0 | 5.01 | 5.08 | 4.96 |

### omp_blocked: 1024×1024, K=3

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.60 | 1.99 | 2.00 |
| 4.0 | 1.44 | 3.75 | 3.85 |
| 8.0 | 2.14 | 3.97 | 4.14 |

### omp_blocked: 1024×1024, K=7

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.97 | 2.00 | 1.99 |
| 4.0 | 2.87 | 3.93 | 3.96 |
| 8.0 | 4.67 | 5.65 | 5.57 |

### omp_blocked: 2048×2048, K=3

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.83 | 2.01 | 2.01 |
| 4.0 | 1.82 | 3.91 | 3.96 |
| 8.0 | 2.70 | 4.41 | 4.46 |

### omp_blocked: 2048×2048, K=7

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.98 | 1.99 | 2.01 |
| 4.0 | 3.22 | 3.92 | 3.98 |
| 8.0 | 5.32 | 5.93 | 5.88 |

### omp_simd: 1024×1024, K=3

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.28 | 2.00 | 1.97 |
| 4.0 | 1.01 | 3.45 | 3.75 |
| 8.0 | 1.39 | 3.51 | 3.93 |

### omp_simd: 1024×1024, K=7

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.44 | 2.00 | 1.99 |
| 4.0 | 1.24 | 3.80 | 3.90 |
| 8.0 | 1.44 | 3.94 | 4.09 |

### omp_simd: 2048×2048, K=3

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.58 | 2.04 | 2.04 |
| 4.0 | 1.52 | 3.57 | 4.48 |
| 8.0 | 1.69 | 3.55 | 4.17 |

### omp_simd: 2048×2048, K=7

| threads | dynamic | guided | static |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.67 | 2.00 | 1.99 |
| 4.0 | 1.65 | 3.85 | 4.01 |
| 8.0 | 1.85 | 3.66 | 3.72 |

## OpenMP parallel efficiency at 8 threads (%)

| config | dynamic | guided | static |
|---|---|---|---|
| omp 1024×1024 K=3 | 44 | 49 | 49 |
| omp 1024×1024 K=7 | 61 | 64 | 62 |
| omp 2048×2048 K=3 | 47 | 49 | 50 |
| omp 2048×2048 K=7 | 63 | 63 | 62 |
| omp_blocked 1024×1024 K=3 | 27 | 50 | 52 |
| omp_blocked 1024×1024 K=7 | 58 | 71 | 70 |
| omp_blocked 2048×2048 K=3 | 34 | 55 | 56 |
| omp_blocked 2048×2048 K=7 | 66 | 74 | 73 |
| omp_simd 1024×1024 K=3 | 17 | 44 | 49 |
| omp_simd 1024×1024 K=7 | 18 | 49 | 51 |
| omp_simd 2048×2048 K=3 | 21 | 44 | 52 |
| omp_simd 2048×2048 K=7 | 23 | 46 | 47 |

## CUDA kernel time by thread block (ms)


### 1024×1024, K=3

| block | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const |
|---|---|---|---|---|---|---|---|---|
| 16×16 | 0.0226 | 0.0216 | 0.0184 | 0.0174 | 0.0236 | 0.0217 | 0.0307 | 0.0287 |
| 1×128 | 0.1551 | 0.1526 | 0.1219 | 0.1251 | 0.0804 | 0.0799 | 0.1096 | 0.1116 |
| 32×16 | 0.0246 | 0.0236 | 0.0195 | 0.0179 | 0.0256 | 0.0236 | 0.0328 | 0.0297 |
| 32×8 | 0.0227 | 0.0225 | 0.0185 | 0.0174 | 0.0242 | 0.0225 | 0.0307 | 0.0287 |
| 8×8 | 0.0246 | 0.0236 | 0.0228 | 0.0212 | 0.0230 | 0.0225 | 0.0315 | 0.0297 |

### 1024×1024, K=7

| block | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const |
|---|---|---|---|---|---|---|---|---|
| 16×16 | 0.0753 | 0.0604 | 0.0625 | 0.0522 | 0.0656 | 0.0492 | 0.0840 | 0.0594 |
| 1×128 | 0.6503 | 0.6630 | 0.5361 | 0.5171 | 0.2063 | 0.1966 | 0.1669 | 0.1536 |
| 32×16 | 0.0748 | 0.0625 | 0.0666 | 0.0553 | 0.0713 | 0.0532 | 0.0860 | 0.0625 |
| 32×8 | 0.0753 | 0.0604 | 0.0671 | 0.0542 | 0.0701 | 0.0522 | 0.0860 | 0.0625 |
| 8×8 | 0.0942 | 0.0768 | 0.0881 | 0.0788 | 0.0655 | 0.0502 | 0.0850 | 0.0614 |

### 2048×2048, K=3

| block | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const |
|---|---|---|---|---|---|---|---|---|
| 16×16 | 0.0815 | 0.0772 | 0.0538 | 0.0491 | 0.0758 | 0.0686 | 0.1075 | 0.1003 |
| 1×128 | 0.6209 | 0.6236 | 0.4741 | 0.4857 | 0.3104 | 0.3092 | 0.4291 | 0.4424 |
| 32×16 | 0.0881 | 0.0829 | 0.0583 | 0.0522 | 0.0860 | 0.0748 | 0.1157 | 0.1044 |
| 32×8 | 0.0809 | 0.0768 | 0.0549 | 0.0496 | 0.0778 | 0.0707 | 0.1087 | 0.1004 |
| 8×8 | 0.0882 | 0.0815 | 0.0758 | 0.0686 | 0.0778 | 0.0737 | 0.1147 | 0.1075 |

### 2048×2048, K=7

| block | cuda | cuda_const | cuda_multiout | cuda_multiout_const | cuda_multiout_tiled | cuda_multiout_tiled_const | cuda_opt | cuda_tiled_const |
|---|---|---|---|---|---|---|---|---|
| 16×16 | 0.2744 | 0.2242 | 0.2171 | 0.1784 | 0.2263 | 0.1690 | 0.2959 | 0.2140 |
| 1×128 | 2.6633 | 2.7448 | 2.1043 | 2.0353 | 0.7971 | 0.7598 | 0.6482 | 0.5796 |
| 32×16 | 0.2730 | 0.2253 | 0.2212 | 0.1823 | 0.2338 | 0.1736 | 0.3000 | 0.2191 |
| 32×8 | 0.2751 | 0.2214 | 0.2212 | 0.1802 | 0.2339 | 0.1741 | 0.3025 | 0.2202 |
| 8×8 | 0.3451 | 0.2806 | 0.3113 | 0.2775 | 0.2309 | 0.1802 | 0.3027 | 0.2243 |
