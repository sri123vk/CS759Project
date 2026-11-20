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
| 512², K=3 | 0.97 | 1.37 | 1.41 | 1.34 | 0.88 |
| 512², K=5 | 0.92 | 1.22 | 1.44 | 1.45 | 0.95 |
| 512², K=7 | 0.90 | 1.04 | 1.46 | 1.48 | 0.88 |
| 1024², K=3 | 0.97 | 1.51 | 1.41 | 1.25 | 0.88 |
| 1024², K=5 | 0.92 | 1.34 | 1.44 | 1.45 | 0.94 |
| 1024², K=7 | 0.90 | 1.11 | 1.52 | 1.53 | 0.88 |
| 2048², K=3 | 0.98 | 1.57 | 1.40 | 1.36 | 0.88 |
| 2048², K=5 | 0.92 | 1.38 | 1.52 | 1.48 | 0.90 |
| 2048², K=7 | 0.90 | 1.09 | 1.52 | 1.60 | 0.90 |

## Speedup of `avx2` over `sse2`, 16 thread(s)

Above 1 means the avx2 build is faster. seq is always single-threaded.

| workload | omp | omp_blocked | omp_simd | omp_tiled | seq |
|---|---|---|---|---|---|
| 512², K=3 | 0.94 | 1.19 | 1.45 | 1.42 | 0.88 |
| 512², K=5 | 1.00 | 1.22 | 1.35 | 1.49 | 0.95 |
| 512², K=7 | 0.98 | 1.23 | 1.45 | 1.52 | 0.88 |
| 1024², K=3 | 1.01 | 1.32 | 1.19 | 1.33 | 0.88 |
| 1024², K=5 | 0.99 | 1.29 | 1.48 | 1.44 | 0.94 |
| 1024², K=7 | 0.97 | 1.28 | 1.51 | 1.54 | 0.88 |
| 2048², K=3 | 0.96 | 1.45 | 1.46 | 1.34 | 0.88 |
| 2048², K=5 | 0.94 | 1.37 | 1.53 | 1.62 | 0.90 |
| 2048², K=7 | 0.94 | 1.38 | 1.56 | 1.87 | 0.90 |

## Speedup of `native` over `sse2`, 1 thread(s)

Above 1 means the native build is faster. seq is always single-threaded.

| workload | omp | omp_blocked | omp_simd | omp_tiled | seq |
|---|---|---|---|---|---|
| 512², K=3 | 0.91 | 1.29 | 1.24 | 1.46 | 0.83 |
| 512², K=5 | 0.67 | 0.96 | 1.27 | 1.34 | 0.67 |
| 512², K=7 | 0.73 | 0.89 | 1.28 | 1.34 | 0.71 |
| 1024², K=3 | 0.91 | 1.43 | 1.35 | 1.28 | 0.83 |
| 1024², K=5 | 0.67 | 1.06 | 1.36 | 1.26 | 0.67 |
| 1024², K=7 | 0.72 | 0.94 | 1.44 | 1.37 | 0.69 |
| 2048², K=3 | 0.90 | 1.48 | 1.32 | 1.53 | 0.82 |
| 2048², K=5 | 0.66 | 1.09 | 1.41 | 1.37 | 0.67 |
| 2048², K=7 | 0.73 | 0.94 | 1.41 | 1.50 | 0.72 |

## Speedup of `native` over `sse2`, 16 thread(s)

Above 1 means the native build is faster. seq is always single-threaded.

| workload | omp | omp_blocked | omp_simd | omp_tiled | seq |
|---|---|---|---|---|---|
| 512², K=3 | 0.93 | 1.19 | 1.35 | 1.44 | 0.83 |
| 512², K=5 | 0.72 | 1.09 | 1.25 | 1.37 | 0.67 |
| 512², K=7 | 0.74 | 1.08 | 1.30 | 1.36 | 0.71 |
| 1024², K=3 | 0.94 | 1.31 | 1.36 | 1.09 | 0.83 |
| 1024², K=5 | 0.70 | 1.22 | 1.38 | 1.13 | 0.67 |
| 1024², K=7 | 0.78 | 1.20 | 1.42 | 1.26 | 0.69 |
| 2048², K=3 | 0.90 | 1.42 | 1.35 | 1.19 | 0.82 |
| 2048², K=5 | 0.66 | 1.32 | 1.47 | 1.39 | 0.67 |
| 2048², K=7 | 0.76 | 1.33 | 1.52 | 1.59 | 0.72 |

## Median time (ms) at 2048×2048

| build | omp t1 | omp t16 | omp_blocked t1 | omp_blocked t16 | omp_simd t1 | omp_simd t16 | omp_tiled t1 | omp_tiled t16 | seq t1 |
|---|---|---|---|---|---|---|---|---|---|
| sse2, K=3 | 33.49 | 4.19 | 4.96 | 0.52 | 3.73 | 0.48 | 3.78 | 0.44 | 30.32 |
| sse2, K=5 | 53.51 | 6.50 | 10.35 | 1.12 | 9.71 | 1.28 | 7.05 | 0.84 | 53.98 |
| sse2, K=7 | 114.40 | 12.49 | 18.71 | 2.07 | 18.16 | 2.63 | 12.21 | 1.65 | 114.94 |
| avx2, K=3 | 34.24 | 4.35 | 3.15 | 0.36 | 2.66 | 0.33 | 2.77 | 0.33 | 34.48 |
| avx2, K=5 | 57.88 | 6.92 | 7.50 | 0.82 | 6.37 | 0.84 | 4.76 | 0.52 | 59.73 |
| avx2, K=7 | 127.26 | 13.22 | 17.19 | 1.50 | 11.92 | 1.69 | 7.63 | 0.88 | 127.32 |
| native, K=3 | 37.03 | 4.66 | 3.36 | 0.37 | 2.83 | 0.35 | 2.47 | 0.37 | 36.80 |
| native, K=5 | 80.64 | 9.84 | 9.53 | 0.85 | 6.87 | 0.88 | 5.13 | 0.61 | 81.13 |
| native, K=7 | 157.34 | 16.35 | 19.91 | 1.56 | 12.86 | 1.73 | 8.13 | 1.03 | 158.63 |

## Loops GCC vectorized (`-fopt-info-vec-optimized`)

"in-order sum": GCC vectorized only the multiplies; the additions into the
single float accumulator still run one at a time (fold-left reduction), because
reordering them would change the rounding without `-ffast-math`. Such a loop
gains little, and it only runs K times per pixel here.

| source:line | what | sse2 | avx2 | native |
|---|---|---|---|---|
| conv2d_omp.cpp:25 | basic block part | no | no | 16 B (4 floats) |
| conv2d_omp.cpp:40 | loop | 16 B (4 floats), in-order sum | 32 B (8 floats), in-order sum | 64 B (16 floats), in-order sum |
| conv2d_omp_blocked.cpp:27 | loop | 16 B (4 floats), in-order sum | 32 B (8 floats), in-order sum | 64 B (16 floats), in-order sum |
| conv2d_omp_blocked.cpp:55 | basic block part | 16 B (4 floats) | 16 B (4 floats) | 16 B (4 floats) |
| conv2d_omp_blocked.cpp:76 | loop | 16 B (4 floats) | 32 B (8 floats) | 64 B (16 floats) |
| conv2d_omp_simd.cpp:24 | basic block part | 16 B (4 floats) | 16 B (4 floats) | 16 B (4 floats) |
| conv2d_omp_simd.cpp:47 | loop | 16 B (4 floats) | 32 B (8 floats) | 64 B (16 floats) |
| conv2d_omp_tiled.cpp:41 | loop | 16 B (4 floats), in-order sum | 32 B (8 floats), in-order sum | 64 B (16 floats), in-order sum |
| conv2d_omp_tiled.cpp:63 | loop | 16 B (4 floats) | 32 B (8 floats) | 64 B (16 floats) |
| conv2d_omp_tiled.cpp:90 | loop | 16 B (4 floats) | 32 B (8 floats) | 64 B (16 floats) |
| conv2d_omp_tiled.cpp:115 | basic block part | 16 B (4 floats) | 16 B (4 floats) | 16 B (4 floats) |
| conv2d_seq.cpp:25 | loop | 16 B (4 floats), in-order sum | 32 B (8 floats), in-order sum | 64 B (16 floats), in-order sum |

