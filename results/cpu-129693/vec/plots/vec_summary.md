# Compiler vectorization study

Instruction sets each build enables on the benchmark machine:

```
# sse2: g++ -march=x86-64 on euler01.engr.wisc.edu
-march=                     		x86-64
-mavx                       		[disabled]
-mavx2                      		[disabled]
-mavx512f                   		[disabled]
-mavx512vl                  		[disabled]
-mfma                       		[disabled]
-msse4.2                    		[disabled]

# avx2: g++ -march=x86-64-v3 on euler01.engr.wisc.edu
-march=                     		x86-64-v3
-mavx                       		[enabled]
-mavx2                      		[enabled]
-mavx512f                   		[disabled]
-mavx512vl                  		[disabled]
-mfma                       		[enabled]
-msse4.2                    		[enabled]

# native: g++ -march=native on euler01.engr.wisc.edu
-march=                     		znver4
-mavx                       		[enabled]
-mavx2                      		[enabled]
-mavx512f                   		[enabled]
-mavx512vl                  		[enabled]
-mfma                       		[enabled]
-msse4.2                    		[enabled]
```

## Speedup of `avx2` over `sse2`, 1 thread(s)

Above 1 means the avx2 build is faster. seq is always single-threaded.

| workload | omp | omp_blocked | omp_simd | omp_tiled | seq |
|---|---|---|---|---|---|
| 512², K=3 | 0.95 | 1.28 | 1.42 | 1.34 | 0.98 |
| 512², K=5 | 0.92 | 1.23 | 1.42 | 1.46 | 0.96 |
| 512², K=7 | 0.89 | 1.02 | 1.47 | 1.43 | 0.93 |
| 1024², K=3 | 0.95 | 1.41 | 1.37 | 1.30 | 0.98 |
| 1024², K=5 | 0.92 | 1.31 | 1.42 | 1.44 | 0.97 |
| 1024², K=7 | 0.90 | 1.11 | 1.49 | 1.52 | 0.93 |
| 2048², K=3 | 0.96 | 1.52 | 1.39 | 1.34 | 0.98 |
| 2048², K=5 | 0.93 | 1.37 | 1.41 | 1.52 | 0.97 |
| 2048², K=7 | 0.89 | 1.09 | 1.48 | 1.63 | 0.93 |

## Speedup of `avx2` over `sse2`, 16 thread(s)

Above 1 means the avx2 build is faster. seq is always single-threaded.

| workload | omp | omp_blocked | omp_simd | omp_tiled | seq |
|---|---|---|---|---|---|
| 512², K=3 | 0.97 | 1.17 | 1.34 | 1.40 | 0.98 |
| 512², K=5 | 0.96 | 1.19 | 1.42 | 1.50 | 0.96 |
| 512², K=7 | 0.99 | 1.27 | 1.43 | 1.47 | 0.93 |
| 1024², K=3 | 0.91 | 1.25 | 1.40 | 1.32 | 0.98 |
| 1024², K=5 | 0.96 | 1.28 | 1.49 | 1.42 | 0.97 |
| 1024², K=7 | 0.97 | 1.30 | 1.55 | 1.63 | 0.93 |
| 2048², K=3 | 0.93 | 1.39 | 1.40 | 1.40 | 0.98 |
| 2048², K=5 | 0.98 | 1.24 | 1.50 | 1.58 | 0.97 |
| 2048², K=7 | 0.98 | 1.26 | 1.50 | 1.72 | 0.93 |

## Speedup of `native` over `sse2`, 1 thread(s)

Above 1 means the native build is faster. seq is always single-threaded.

| workload | omp | omp_blocked | omp_simd | omp_tiled | seq |
|---|---|---|---|---|---|
| 512², K=3 | 0.92 | 1.23 | 1.24 | 1.44 | 0.94 |
| 512², K=5 | 0.67 | 0.95 | 1.26 | 1.34 | 0.68 |
| 512², K=7 | 0.69 | 0.83 | 1.30 | 1.29 | 0.73 |
| 1024², K=3 | 0.92 | 1.37 | 1.28 | 1.33 | 0.95 |
| 1024², K=5 | 0.66 | 1.05 | 1.30 | 1.30 | 0.68 |
| 1024², K=7 | 0.70 | 0.89 | 1.37 | 1.37 | 0.73 |
| 2048², K=3 | 0.93 | 1.46 | 1.27 | 1.48 | 0.94 |
| 2048², K=5 | 0.66 | 1.09 | 1.29 | 1.39 | 0.68 |
| 2048², K=7 | 0.70 | 0.90 | 1.35 | 1.50 | 0.73 |

## Speedup of `native` over `sse2`, 16 thread(s)

Above 1 means the native build is faster. seq is always single-threaded.

| workload | omp | omp_blocked | omp_simd | omp_tiled | seq |
|---|---|---|---|---|---|
| 512², K=3 | 0.88 | 1.16 | 1.17 | 1.36 | 0.94 |
| 512², K=5 | 0.68 | 1.02 | 1.25 | 1.33 | 0.68 |
| 512², K=7 | 0.73 | 1.06 | 1.29 | 1.29 | 0.73 |
| 1024², K=3 | 0.93 | 1.29 | 1.28 | 1.13 | 0.95 |
| 1024², K=5 | 0.67 | 1.17 | 1.34 | 1.23 | 0.68 |
| 1024², K=7 | 0.73 | 1.20 | 1.47 | 1.38 | 0.73 |
| 2048², K=3 | 0.91 | 1.45 | 1.38 | 1.32 | 0.94 |
| 2048², K=5 | 0.68 | 1.23 | 1.46 | 1.38 | 0.68 |
| 2048², K=7 | 0.74 | 1.32 | 1.57 | 1.49 | 0.73 |

## Median time (ms) at 2048×2048

| build | omp t1 | omp t16 | omp_blocked t1 | omp_blocked t16 | omp_simd t1 | omp_simd t16 | omp_tiled t1 | omp_tiled t16 | seq t1 |
|---|---|---|---|---|---|---|---|---|---|
| sse2, K=3 | 43.33 | 5.76 | 5.52 | 0.59 | 3.82 | 0.48 | 4.14 | 0.46 | 44.16 |
| sse2, K=5 | 64.37 | 8.79 | 10.60 | 1.17 | 9.18 | 1.36 | 7.40 | 0.88 | 64.77 |
| sse2, K=7 | 128.78 | 16.44 | 18.89 | 2.32 | 17.83 | 2.76 | 12.73 | 1.66 | 130.53 |
| avx2, K=3 | 45.37 | 6.18 | 3.63 | 0.42 | 2.74 | 0.35 | 3.09 | 0.33 | 45.19 |
| avx2, K=5 | 69.50 | 8.97 | 7.74 | 0.95 | 6.50 | 0.90 | 4.86 | 0.56 | 67.03 |
| avx2, K=7 | 143.92 | 16.74 | 17.26 | 1.84 | 12.02 | 1.85 | 7.79 | 0.97 | 141.10 |
| native, K=3 | 46.65 | 6.34 | 3.79 | 0.40 | 3.00 | 0.35 | 2.80 | 0.35 | 46.96 |
| native, K=5 | 97.61 | 12.86 | 9.70 | 0.95 | 7.12 | 0.93 | 5.32 | 0.64 | 95.49 |
| native, K=7 | 183.62 | 22.26 | 21.10 | 1.76 | 13.25 | 1.76 | 8.50 | 1.12 | 178.50 |

## Loops GCC vectorized (`-fopt-info-vec-optimized`)

"in-order sum": GCC vectorized only the multiplies; the additions into the
single float accumulator still run one at a time (fold-left reduction), because
reordering them would change the rounding without `-ffast-math`. Such a loop
gains little, and it only runs K times per pixel here.

| source:line | what | sse2 | avx2 | native |
|---|---|---|---|---|
| omp/conv2d_omp.cpp:28 | basic block part | 16 B (4 floats) | 16 B (4 floats) | 16 B (4 floats) |
| omp/conv2d_omp.cpp:46 | loop | 16 B (4 floats), in-order sum | 32 B (8 floats), in-order sum | 64 B (16 floats), in-order sum |
| omp/conv2d_omp_blocked.cpp:37 | loop | 16 B (4 floats), in-order sum | 32 B (8 floats), in-order sum | 64 B (16 floats), in-order sum |
| omp/conv2d_omp_blocked.cpp:69 | basic block part | 16 B (4 floats) | 16 B (4 floats) | 16 B (4 floats) |
| omp/conv2d_omp_blocked.cpp:93 | loop | 16 B (4 floats) | 32 B (8 floats) | 64 B (16 floats) |
| omp/conv2d_omp_simd.cpp:31 | basic block part | 16 B (4 floats) | 16 B (4 floats) | 16 B (4 floats) |
| omp/conv2d_omp_simd.cpp:60 | loop | 16 B (4 floats) | 32 B (8 floats) | 64 B (16 floats) |
| omp/conv2d_omp_tiled.cpp:49 | loop | 16 B (4 floats), in-order sum | 32 B (8 floats), in-order sum | 64 B (16 floats), in-order sum |
| omp/conv2d_omp_tiled.cpp:72 | loop | 16 B (4 floats) | 32 B (8 floats) | 64 B (16 floats) |
| omp/conv2d_omp_tiled.cpp:101 | loop | 16 B (4 floats) | 32 B (8 floats) | 64 B (16 floats) |
| omp/conv2d_omp_tiled.cpp:129 | basic block part | 16 B (4 floats) | 16 B (4 floats) | 16 B (4 floats) |
| seq/conv2d_seq.cpp:35 | loop | 16 B (4 floats), in-order sum | 32 B (8 floats), in-order sum | 64 B (16 floats), in-order sum |

