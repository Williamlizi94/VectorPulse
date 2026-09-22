# Persistent GPU database: measured results

Hardware: Intel Core i9-14900F (24 cores, `hardware_concurrency()=32`), NVIDIA GeForce RTX 4080 SUPER (16 GB, compute 8.9), driver 610.60. Windows WDDM, shared display GPU. CUDA 12.9 Update 1 / nvcc 12.9.86, static runtime, MSVC 19.44.35228.0, Release. Same build flags as [Milestone 3 environment](milestone3-environment.txt): `--fmad=false`, sm_89 plus compute_89 PTX, no fast math or kernel changes.

All runs use dimension 768, K=10, seed-42 identical inputs across backends, one full warm-up round and three measured repetitions with seed-2026 shuffled backend order. 1K/10K/100K use 100 queries (300 measured); 1M uses 10 queries (30 measured) to bound runtime. CPU thread counts are 1,2,4,8,16,32, except 1M uses 16,32 to bound simultaneous host storage. Best CPU means best among these measured variants, not an exhaustive optimum.

The persistent index is built before warm-up. Build cost is reported separately below; steady-state end-to-end includes query validation/norm, cache lookup, scratch allocation, query upload, unchanged kernel, score download, CPU Top-K, and scratch cleanup. Lazy rebuilds in normal API use are included in query total. Each steady-state persistent query was checked to perform no rebuild and transfer exactly 3072 query bytes and zero database bytes.

## End-to-end comparison

| Vectors | Naive CUDA ms | Persistent CUDA ms | Speedup vs naive | Best CPU | CPU ms | Persistent speedup vs best CPU |
|---|---:|---:|---:|---|---:|---:|
| 1k | 1.684 | 0.610 | 2.76x | AVX2 | 0.131 | 0.21x |
| 10k | 16.946 | 3.792 | 4.47x | MultithreadedAVX2[4] | 1.055 | 0.28x |
| 100k | 103.011 | 4.789 | 21.51x | MultithreadedAVX2[32] | 8.356 | 1.74x |
| 1m | 963.108 | 37.492 | 25.69x | MultithreadedAVX2[32] | 58.945 | 1.57x |

Speedups below 1 mean persistent CUDA is slower. Comparisons use paired outer end-to-end means, never kernel-only times.

## Mean CUDA phases (ms/query)

| Vectors | Mode | Flatten + norm | H2D | Kernel | D2H | CPU Top-K | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|
| 1k | CUDA-Naive | 0.535410 | 0.439750 | 0.283474 | 0.011835 | 0.011919 | 1.684 |
| 1k | CUDA-Persistent | 0.000347 | 0.154178 | 0.280625 | 0.012647 | 0.011339 | 0.610 |
| 10k | CUDA-Naive | 5.885723 | 7.158762 | 1.371213 | 0.108242 | 0.161822 | 16.946 |
| 10k | CUDA-Persistent | 0.000352 | 1.236191 | 1.339757 | 0.148110 | 0.138928 | 3.792 |
| 100k | CUDA-Naive | 57.943172 | 30.572097 | 3.787528 | 0.078547 | 1.258301 | 103.011 |
| 100k | CUDA-Persistent | 0.000321 | 0.287172 | 2.875142 | 0.094780 | 1.222959 | 4.789 |
| 1m | CUDA-Naive | 583.836910 | 263.631130 | 14.412398 | 0.342807 | 14.448023 | 963.108 |
| 1m | CUDA-Persistent | 0.000333 | 0.176247 | 22.393323 | 0.337873 | 13.369360 | 37.492 |

H2D in naive mode includes the full matrix plus query; persistent H2D includes only the query. Persistent flatten+norm is query norm only. H2D is synchronized host elapsed time and can include WDDM/default-stream/device wait overhead, not simply bytes divided by bandwidth. Kernel timings use CUDA events. D2H is synchronous host elapsed time. CPU Top-K includes finite-score checks and result materialization. Allocation, device/event management and cleanup are in end-to-end but outside individual phases; phase sums need not equal total.

## One-time persistent index build

| Vectors | Database bytes | Flatten ms | H2D ms | Total ms |
|---|---:|---:|---:|---:|
| 1k | 3072000 | 0.576400 | 0.692500 | 1.515100 |
| 10k | 30720000 | 5.752300 | 3.118800 | 9.835000 |
| 100k | 307200000 | 57.213000 | 25.103500 | 89.760000 |
| 1m | 3072000000 | 822.965700 | 466.172800 | 1456.554900 |

Build totals include allocation, synchronized upload and host scratch destruction. These are single build observations, not repeated distributions. Rebuilds retain the old GPU allocation until a replacement succeeds and can require roughly two database matrices of VRAM.

## Full end-to-end distributions

| Vectors | Backend | Mean ms | Median ms | Min ms | Max ms | QPS |
|---|---|---:|---:|---:|---:|---:|
| 1k | Scalar | 0.599 | 0.594 | 0.550 | 0.710 | 1668.377 |
| 1k | AVX2 | 0.131 | 0.128 | 0.110 | 0.334 | 7656.382 |
| 1k | MultithreadedScalar[1] | 0.608 | 0.590 | 0.545 | 2.337 | 1645.714 |
| 1k | MultithreadedScalar[2] | 0.385 | 0.378 | 0.354 | 0.549 | 2596.411 |
| 1k | MultithreadedScalar[4] | 0.286 | 0.276 | 0.247 | 0.586 | 3491.661 |
| 1k | MultithreadedScalar[8] | 0.348 | 0.335 | 0.278 | 0.581 | 2872.988 |
| 1k | MultithreadedScalar[16] | 0.629 | 0.608 | 0.510 | 1.112 | 1588.756 |
| 1k | MultithreadedScalar[32] | 1.197 | 1.168 | 1.028 | 1.744 | 835.734 |
| 1k | MultithreadedAVX2[1] | 0.132 | 0.126 | 0.120 | 0.310 | 7555.666 |
| 1k | MultithreadedAVX2[2] | 0.141 | 0.132 | 0.115 | 0.429 | 7093.943 |
| 1k | MultithreadedAVX2[4] | 0.177 | 0.170 | 0.138 | 0.391 | 5638.133 |
| 1k | MultithreadedAVX2[8] | 0.322 | 0.306 | 0.269 | 2.081 | 3110.049 |
| 1k | MultithreadedAVX2[16] | 0.606 | 0.588 | 0.507 | 1.135 | 1649.121 |
| 1k | MultithreadedAVX2[32] | 1.177 | 1.151 | 0.979 | 1.614 | 849.289 |
| 1k | CUDA-Naive | 1.684 | 1.660 | 1.499 | 2.203 | 593.828 |
| 1k | CUDA-Persistent | 0.610 | 0.580 | 0.540 | 1.074 | 1638.756 |
| 10k | Scalar | 6.384 | 6.267 | 5.999 | 16.840 | 156.632 |
| 10k | AVX2 | 2.811 | 2.665 | 2.324 | 33.844 | 355.714 |
| 10k | MultithreadedScalar[1] | 6.415 | 6.349 | 6.081 | 11.136 | 155.893 |
| 10k | MultithreadedScalar[2] | 3.720 | 3.461 | 3.309 | 16.650 | 268.793 |
| 10k | MultithreadedScalar[4] | 2.140 | 1.969 | 1.722 | 4.814 | 467.236 |
| 10k | MultithreadedScalar[8] | 1.905 | 1.456 | 1.063 | 6.278 | 524.866 |
| 10k | MultithreadedScalar[16] | 2.332 | 1.523 | 1.171 | 25.072 | 428.727 |
| 10k | MultithreadedScalar[32] | 2.267 | 1.927 | 1.471 | 5.329 | 441.085 |
| 10k | MultithreadedAVX2[1] | 2.748 | 2.689 | 2.467 | 16.570 | 363.870 |
| 10k | MultithreadedAVX2[2] | 1.552 | 1.444 | 1.185 | 4.301 | 644.373 |
| 10k | MultithreadedAVX2[4] | 1.055 | 0.913 | 0.763 | 3.558 | 947.897 |
| 10k | MultithreadedAVX2[8] | 1.239 | 0.819 | 0.579 | 6.808 | 807.025 |
| 10k | MultithreadedAVX2[16] | 1.729 | 1.044 | 0.784 | 6.827 | 578.481 |
| 10k | MultithreadedAVX2[32] | 2.222 | 1.551 | 1.150 | 9.920 | 450.107 |
| 10k | CUDA-Naive | 16.946 | 17.522 | 10.283 | 40.217 | 59.011 |
| 10k | CUDA-Persistent | 3.792 | 3.662 | 0.711 | 25.458 | 263.739 |
| 100k | Scalar | 63.007 | 60.107 | 58.278 | 310.126 | 15.871 |
| 100k | AVX2 | 27.561 | 26.127 | 25.405 | 292.373 | 36.283 |
| 100k | MultithreadedScalar[1] | 64.848 | 61.743 | 59.372 | 349.965 | 15.421 |
| 100k | MultithreadedScalar[2] | 32.964 | 32.297 | 30.374 | 53.720 | 30.336 |
| 100k | MultithreadedScalar[4] | 18.329 | 18.155 | 15.273 | 27.907 | 54.558 |
| 100k | MultithreadedScalar[8] | 13.408 | 13.184 | 10.064 | 33.002 | 74.582 |
| 100k | MultithreadedScalar[16] | 12.028 | 11.456 | 8.297 | 172.949 | 83.141 |
| 100k | MultithreadedScalar[32] | 10.241 | 8.328 | 6.984 | 185.028 | 97.647 |
| 100k | MultithreadedAVX2[1] | 27.766 | 26.328 | 25.594 | 145.352 | 36.015 |
| 100k | MultithreadedAVX2[2] | 15.469 | 14.687 | 13.988 | 37.374 | 64.645 |
| 100k | MultithreadedAVX2[4] | 10.229 | 9.678 | 8.472 | 22.893 | 97.761 |
| 100k | MultithreadedAVX2[8] | 9.124 | 8.639 | 7.086 | 28.046 | 109.606 |
| 100k | MultithreadedAVX2[16] | 9.509 | 8.145 | 6.814 | 247.214 | 105.166 |
| 100k | MultithreadedAVX2[32] | 8.356 | 7.427 | 6.362 | 22.393 | 119.677 |
| 100k | CUDA-Naive | 103.011 | 97.194 | 90.436 | 270.793 | 9.708 |
| 100k | CUDA-Persistent | 4.789 | 3.240 | 2.941 | 79.732 | 208.793 |
| 1m | Scalar | 624.835 | 611.540 | 596.678 | 891.911 | 1.600 |
| 1m | AVX2 | 262.597 | 259.332 | 256.548 | 280.995 | 3.808 |
| 1m | MultithreadedScalar[16] | 88.655 | 86.869 | 80.947 | 101.719 | 11.280 |
| 1m | MultithreadedScalar[32] | 73.042 | 69.406 | 67.266 | 105.347 | 13.691 |
| 1m | MultithreadedAVX2[16] | 66.406 | 65.622 | 63.408 | 74.615 | 15.059 |
| 1m | MultithreadedAVX2[32] | 58.945 | 58.425 | 57.618 | 64.298 | 16.965 |
| 1m | CUDA-Naive | 963.108 | 944.515 | 905.838 | 1176.103 | 1.038 |
| 1m | CUDA-Persistent | 37.492 | 29.335 | 28.801 | 271.012 | 26.673 |

## Correctness and checksums

Every warm-up and measured result matched scalar IDs/ranks and score tolerance 1e-6. All backends within each run produced identical checksums. Maximum measured score error was zero.

| Vectors | Score checksum | ID checksum |
|---|---:|---:|
| 1k | 287.063861355 | 11688052443322439949 |
| 10k | 360.302542560 | 6942403294749163753 |
| 100k | 425.896235436 | 4652264718457849833 |
| 1m | 48.008113995 | 10511493593234606401 |

CUDA-enabled Release: 288 tests passed, zero skipped/failed. CPU-only Release: 243 passed, 45 hardware tests skipped, zero failures. Pre-change CUDA build: 266 passed. Tests include both storage modes, dimension tails, K edge cases, ties, extreme finite values, repeated use/rebuilds, rejected inserts and concurrent initial readers. [Kernel/CPU preservation hashes](persistent-preservation.txt) confirm unchanged kernel text and 17 CPU source/header files.

## Commands and raw logs

Build commands remain in [README](../../README.md#build-and-test). Benchmarks were run sequentially after builds/tests:

```powershell
./out/build-cuda/vectorpulse_benchmark.exe --vectors 1000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 1,2,4,8,16,32
./out/build-cuda/vectorpulse_benchmark.exe --vectors 10000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 1,2,4,8,16,32
./out/build-cuda/vectorpulse_benchmark.exe --vectors 100000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 1,2,4,8,16,32
./out/build-cuda/vectorpulse_benchmark.exe --vectors 1000000 --dimension 768 --queries 10 --top-k 10 --warmups 1 --repetitions 3 --threads 16,32
```

Raw logs: [1K](persistent-1k.txt), [10K](persistent-10k.txt), [100K](persistent-100k.txt), [1M](persistent-1m.txt). Logs retain every round and all phase mean/median/min/max values. [CUDA tests](persistent-cuda-tests.txt), [CPU-only tests](persistent-cpu-tests.txt).

## Files changed for this milestone

- `include/vectorpulse/cuda_search_backend.h`: persistent/default and per-query modes, build API, immutable index ownership and per-call diagnostics.
- `src/cuda_search_backend.cpp`: serialized transactional cache construction and mode selection.
- `src/cuda_runtime_bridge.h`, `src/cuda_runtime.cu`, `src/cuda_runtime_stub.cpp`: opaque RAII index, one-time matrix upload, query-only execution, and CPU-only stubs.
- `tests/cuda_search_backend_test.cpp`: both-mode scalar comparisons and persistent lifecycle/update/concurrency checks.
- `benchmarks/scalar_benchmark.cpp`: paired modes, explicit build cost and steady-state transfer assertions.
- `README.md`, `docs/architecture.md`, and `docs/benchmarks/persistent-*`: documentation, measured reports and raw validation evidence.

Existing CMake and other uncommitted milestone changes were retained; this milestone did not change CMake, CPU backends, VectorStore or the naive kernel.

## Limits and interpretation

Persistent CUDA was 2.76x/4.47x/21.51x/25.69x faster than the paired naive path at 1K/10K/100K/1M. CPU remained faster at 1K/10K; persistent CUDA was 1.74x and 1.57x faster than the best measured CPU at 100K/1M. The 1M persistent run includes a 271.012 ms maximum versus a 29.335 ms median, which raises its mean to 37.492 ms. No samples were discarded.

Removing full-database flatten/upload reduces steady-state query latency. Remaining costs include the deliberately unchanged strided, FP64-accumulating kernel, device synchronization, query/score allocations, full score download and serial CPU Top-K. No GPU Top-K, kernel optimization, pinned memory, transfer overlap or scratch pooling was added.

No affinity, clock controls, exclusive GPU access or hardware-counter profiling was used. WDDM/display activity and shared-machine load cause variation, especially phase outliers; do not infer pure bandwidth or kernel improvements from these observations. The old naive-only Milestone 3 numbers are historical, not the paired baseline used here. Query sets/sample counts differ across sizes. Rebuild failure handling is transactional but allocation/copy failures were not fault-injected. Inserts require external exclusion from reads/builds; concurrent writers are unsupported. Only this Windows/MSVC/GPU configuration was measured.
