# Milestone 3 measured CPU/CUDA baseline

All times are actual measurements on the recorded machine. CUDA performance
comparisons use end-to-end query latency, never kernel-only latency.

| Vectors | Scalar ms / QPS | AVX2 ms / QPS | Best measured threaded AVX2 (threads): ms / QPS | CUDA kernel ms | CUDA end-to-end ms / QPS | CUDA slowdown vs best CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.693 / 1442.303 | 0.337 / 2967.239 | 1: 0.246 / 4071.390 | 0.291 | 1.883 / 531.026 | 7.67x slower |
| 10,000 | 12.160 / 82.239 | 7.111 / 140.619 | 8: 1.707 / 585.748 | 1.869 | 24.588 / 40.670 | 14.40x slower |
| 100,000 | 121.072 / 8.260 | 71.784 / 13.931 | 16: 9.378 / 106.633 | 11.108 | 207.770 / 4.813 | 22.16x slower |
| 1,000,000 | 1181.797 / 0.846 | 711.810 / 1.405 | 16: 81.027 / 12.342 | 16.877 | 1647.002 / 0.607 | 20.33x slower |

## Sampling and hardware

Dimension 768, Top-K=10, input seed 42, order seed 2026, one warm-up round.
1k and 10k: 100 queries, three measured repetitions, 15 backend variants.
100k: 100 queries, five repetitions, 15 variants.
1M: 10 queries, three repetitions, seven variants (thread counts 16,32 only).
Data/query lists are identical across backends within a run, not across sizes.
All stores coexist; memory footprint affects cache/placement.

CPU: Intel Core i9-14900F, 24 cores, 32 logical threads.
GPU: RTX 4080 SUPER, compute 8.9, 17170956288 bytes global memory.
Driver: 610.60, driver API 13030; runtime API 12090.
Toolkit: CUDA 12.9 Update 1, nvcc 12.9.86, cudart package 12.9.79.
MSVC 19.44.35228.0, Release, default precise host arithmetic.
CUDA target sm_89/compute_89, --fmad=false, no fast math.
No affinity/frequency controls; WDDM/display GPU shared with other applications.

## Full 100k comparison

| Backend, 100k vectors | Mean ms | Median ms | Min ms | Max ms | QPS |
|---|---:|---:|---:|---:|---:|
| Scalar | 121.072 | 117.672 | 76.303 | 289.382 | 8.260 |
| AVX2 | 71.784 | 69.321 | 27.798 | 294.800 | 13.931 |
| MultithreadedScalar[1] | 120.891 | 116.521 | 83.128 | 343.429 | 8.272 |
| MultithreadedScalar[2] | 54.180 | 51.372 | 41.178 | 255.082 | 18.457 |
| MultithreadedScalar[4] | 29.177 | 28.357 | 24.247 | 227.132 | 34.274 |
| MultithreadedScalar[8] | 17.378 | 16.777 | 13.267 | 120.566 | 57.544 |
| MultithreadedScalar[16] | 11.798 | 11.562 | 9.606 | 34.017 | 84.758 |
| MultithreadedScalar[32] | 12.676 | 11.543 | 10.161 | 95.224 | 78.892 |
| MultithreadedAVX2[1] | 73.184 | 69.595 | 26.985 | 283.813 | 13.664 |
| MultithreadedAVX2[2] | 35.290 | 34.245 | 16.593 | 260.111 | 28.337 |
| MultithreadedAVX2[4] | 19.630 | 19.047 | 10.578 | 43.425 | 50.942 |
| MultithreadedAVX2[8] | 13.239 | 13.025 | 10.544 | 26.257 | 75.535 |
| MultithreadedAVX2[16] | 9.378 | 9.064 | 7.766 | 18.557 | 106.633 |
| MultithreadedAVX2[32] | 9.526 | 9.184 | 8.425 | 20.618 | 104.978 |
| CUDA | 207.770 | 199.856 | 150.088 | 439.316 | 4.813 |

## GPU latency distributions by size

| Vectors | CUDA end-to-end mean / median / min / max ms | Kernel-only mean / median / min / max ms |
|---:|---:|---:|
| 1000 | 1.883 / 1.659 / 1.488 / 4.290 | 0.291 / 0.288 / 0.274 / 0.633 |
| 10000 | 24.588 / 23.426 / 18.242 / 40.995 | 1.869 / 1.523 / 1.226 / 7.138 |
| 100000 | 207.770 / 199.856 / 150.088 / 439.316 | 11.108 / 8.765 / 1.421 / 148.516 |
| 1000000 | 1647.002 / 1709.675 / 919.558 / 2007.835 | 16.877 / 15.306 / 13.864 / 28.059 |

## 100k CUDA phase distributions

| CUDA phase, 100k vectors | Mean ms | Median ms | Min ms | Max ms |
|---|---:|---:|---:|---:|
| flatten + query norm (host) | 101.431 | 100.631 | 57.698 | 215.424 |
| H2D vectors + query (host, synchronized) | 74.037 | 67.629 | 39.105 | 135.141 |
| kernel-only (CUDA events) | 11.108 | 8.765 | 1.421 | 148.516 |
| D2H scores (host) | 0.323 | 0.261 | 0.079 | 4.177 |
| CPU Top-K | 1.630 | 1.296 | 1.124 | 9.467 |
| profiled end-to-end | 207.769 | 199.855 | 150.087 | 439.315 |

## Interpretation

No measured GPU crossover occurred. At 100k, CUDA was 22.16x
slower than the best CPU result. Flattening and H2D accounted for about 84.5% of
GPU end-to-end time. Other allocation/cleanup/runtime overhead is included in
total but not isolated. The naive kernel also uses strided per-row access and
double accumulation; potential GPU bandwidth/FP64/scheduling limits are not
separately profiled. Kernel-only time must not stand in for end-to-end performance.

The 1M comparison uses only 30 measured queries per variant and two CPU thread
counts; interpret it as a limited sampled comparison. Observed optimal thread
counts are not a general tuning recommendation. At 1k, the one-thread threaded
backend is fastest by mean but does not exploit multicore parallelism; timing
variation is visible in the individual round logs.

## Checksums and correctness

Every reported variant matched exact scalar IDs/ranks and score tolerance 1e-6,
with zero observed score error and identical aggregate checksums within each run.

| Vectors | Score checksum | ID checksum |
|---:|---:|---:|
| 1000 | 287.063861355 | 11688052443322439949 |
| 10000 | 360.302542560 | 6942403294749163753 |
| 100000 | 709.827059060 | 8705275316278055177 |
| 1000000 | 48.008113995 | 10511493593234606401 |

## Raw logs

- [1000 vectors](milestone3-1000.txt)
- [10000 vectors](milestone3-10000.txt)
- [100000 vectors](milestone3-100000.txt)
- [1000000 vectors](milestone3-1000000.txt)
- [Pre-change baseline](milestone3-baseline.txt)
- [Environment and commands](milestone3-environment.txt)
