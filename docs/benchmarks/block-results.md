# Block-parallel CUDA: measured results

One block of 256 threads processes one vector using two double shared-memory arrays and a full-block binary tree reduction. Both GPU variants use the existing persistent index and identical query upload, score download and CPU Top-K code. The naive kernel remains unchanged. The default kernel remains naive; select `CudaKernel::BlockParallel` with persistent storage for the new implementation. No warp-level optimization, GPU Top-K or CPU redesign was added.

## Hardware and method

Intel Core i9-14900F, 24 cores, `hardware_concurrency()=32`; NVIDIA GeForce RTX 4080 SUPER, 16 GB, compute capability 8.9; driver 610.60, Windows WDDM/shared display GPU. CUDA 12.9 Update 1 / nvcc 12.9.86, MSVC 19.44.35228.0, Release, static CUDA runtime. Existing flags: `--fmad=false`, sm_89 plus compute_89 PTX, no fast math; CPU `/O2 /Ob2`, AVX2 kernel `/arch:AVX2`. See [toolchain details](milestone3-environment.txt).

All sizes use 768 dimensions and K=10. One complete warm-up round precedes three measured repetitions. 1K/10K/100K use 100 queries (300 measured); 1M uses 10 queries (30 measured) to bound runtime. Identical seed-42 data/queries are copied to every backend within a run; backend order is shuffled with seed 2026 each measured round. CPU thread counts are 1,2,4,8,16,32 except at 1M (16,32 to bound host memory). Best CPU means best among the variants measured in that run.

Both persistent indices are built before warm-up; steady-state results exclude initial build cost. Each backend owns its own resident matrix using the same index implementation. Every timed query verifies zero database H2D and exactly 3072 query bytes. End-to-end includes validation/norm, snapshot acquisition, allocations, query H2D, kernel/synchronization, full score D2H, CPU Top-K and scratch teardown. CUDA events measure kernel-only time. Correctness checks, checksums, input preparation and returned-result destruction are outside timing.

## Paired mean latency and speedup

All latency values are milliseconds per query. Speedups are naive/block or CPU/block; below 1 means block-parallel is slower.

| Vectors | Naive kernel | Block kernel | Kernel speedup | Naive end-to-end | Block end-to-end | End-to-end speedup | Best CPU | CPU ms | Block speedup vs CPU |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1k | 0.289812 | 0.039462 | 7.34x | 0.695 | 0.658 | 1.06x | MultithreadedAVX2[2] | 0.335 | 0.51x |
| 10k | 2.147835 | 1.444950 | 1.49x | 6.580 | 5.631 | 1.17x | MultithreadedAVX2[16] | 1.946 | 0.35x |
| 100k | 2.407090 | 3.302910 | 0.73x | 5.179 | 5.977 | 0.87x | MultithreadedAVX2[16] | 9.345 | 1.56x |
| 1m | 34.099332 | 30.769517 | 1.11x | 62.595 | 60.596 | 1.03x | MultithreadedAVX2[16] | 75.824 | 1.25x |

## CUDA timing distributions

Each cell is mean / median / min / max, in milliseconds. No outliers are discarded.

| Vectors | Backend | Kernel | End-to-end |
|---|---|---|---|
| 1k | CUDA-Persistent | 0.289812 / 0.288416 / 0.275456 / 0.676544 | 0.695 / 0.670 / 0.556 / 2.212 |
| 1k | CUDA-BlockParallel | 0.039462 / 0.038192 / 0.023552 / 0.228352 | 0.658 / 0.406 / 0.307 / 6.728 |
| 10k | CUDA-Persistent | 2.147835 / 2.337792 / 0.703488 / 7.991296 | 6.580 / 6.936 / 2.239 / 13.624 |
| 10k | CUDA-BlockParallel | 1.444950 / 1.400832 / 0.414720 / 12.183552 | 5.631 / 5.637 / 2.122 / 17.416 |
| 100k | CUDA-Persistent | 2.407090 / 1.455904 / 1.404928 / 27.873280 | 5.179 / 4.109 / 3.092 / 32.864 |
| 100k | CUDA-BlockParallel | 3.302910 / 1.822720 / 1.671168 / 43.524097 | 5.977 / 3.874 / 3.228 / 49.523 |
| 1m | CUDA-Persistent | 34.099332 / 13.892592 / 13.829120 / 250.474503 | 62.595 / 42.754 / 31.836 / 285.107 |
| 1m | CUDA-BlockParallel | 30.769517 / 17.969152 / 17.059681 / 198.569977 | 60.596 / 50.496 / 38.911 / 225.284 |

## Complete end-to-end CPU/GPU comparison

| Vectors | Backend | Mean ms | Median ms | Min ms | Max ms | QPS |
|---|---|---:|---:|---:|---:|---:|
| 1k | Scalar | 0.731 | 0.613 | 0.576 | 1.153 | 1367.255 |
| 1k | AVX2 | 0.396 | 0.507 | 0.121 | 0.943 | 2524.706 |
| 1k | MultithreadedScalar[1] | 0.727 | 0.598 | 0.556 | 2.350 | 1374.825 |
| 1k | MultithreadedScalar[2] | 0.587 | 0.447 | 0.371 | 2.420 | 1704.146 |
| 1k | MultithreadedScalar[4] | 0.565 | 0.527 | 0.287 | 4.620 | 1769.537 |
| 1k | MultithreadedScalar[8] | 0.736 | 0.688 | 0.345 | 7.950 | 1359.257 |
| 1k | MultithreadedScalar[16] | 1.147 | 1.211 | 0.637 | 2.207 | 871.693 |
| 1k | MultithreadedScalar[32] | 1.936 | 1.735 | 1.359 | 4.503 | 516.531 |
| 1k | MultithreadedAVX2[1] | 0.412 | 0.500 | 0.130 | 4.471 | 2427.291 |
| 1k | MultithreadedAVX2[2] | 0.335 | 0.202 | 0.125 | 2.733 | 2985.485 |
| 1k | MultithreadedAVX2[4] | 0.486 | 0.413 | 0.168 | 3.784 | 2058.559 |
| 1k | MultithreadedAVX2[8] | 0.578 | 0.609 | 0.340 | 0.990 | 1729.367 |
| 1k | MultithreadedAVX2[16] | 1.219 | 0.855 | 0.621 | 14.430 | 820.482 |
| 1k | MultithreadedAVX2[32] | 2.242 | 2.286 | 1.207 | 6.585 | 446.120 |
| 1k | CUDA-Persistent | 0.695 | 0.670 | 0.556 | 2.212 | 1439.313 |
| 1k | CUDA-BlockParallel | 0.658 | 0.406 | 0.307 | 6.728 | 1519.748 |
| 10k | Scalar | 11.393 | 10.733 | 7.001 | 33.275 | 87.769 |
| 10k | AVX2 | 7.012 | 6.575 | 5.835 | 16.729 | 142.617 |
| 10k | MultithreadedScalar[1] | 11.410 | 10.723 | 9.547 | 23.817 | 87.640 |
| 10k | MultithreadedScalar[2] | 6.546 | 6.178 | 5.175 | 12.300 | 152.771 |
| 10k | MultithreadedScalar[4] | 3.618 | 3.446 | 2.907 | 5.973 | 276.409 |
| 10k | MultithreadedScalar[8] | 2.687 | 2.513 | 1.807 | 4.328 | 372.226 |
| 10k | MultithreadedScalar[16] | 2.296 | 2.244 | 1.590 | 5.894 | 435.503 |
| 10k | MultithreadedScalar[32] | 3.030 | 2.971 | 2.397 | 5.158 | 330.013 |
| 10k | MultithreadedAVX2[1] | 7.721 | 6.652 | 6.011 | 124.320 | 129.518 |
| 10k | MultithreadedAVX2[2] | 4.320 | 3.938 | 3.317 | 12.957 | 231.480 |
| 10k | MultithreadedAVX2[4] | 2.623 | 2.417 | 2.000 | 15.853 | 381.218 |
| 10k | MultithreadedAVX2[8] | 1.968 | 1.867 | 1.399 | 19.679 | 508.047 |
| 10k | MultithreadedAVX2[16] | 1.946 | 1.877 | 1.405 | 6.145 | 513.971 |
| 10k | MultithreadedAVX2[32] | 3.208 | 2.728 | 2.355 | 72.475 | 311.709 |
| 10k | CUDA-Persistent | 6.580 | 6.936 | 2.239 | 13.624 | 151.970 |
| 10k | CUDA-BlockParallel | 5.631 | 5.637 | 2.122 | 17.416 | 177.598 |
| 100k | Scalar | 109.385 | 111.337 | 58.095 | 264.594 | 9.142 |
| 100k | AVX2 | 68.655 | 67.895 | 25.697 | 117.171 | 14.566 |
| 100k | MultithreadedScalar[1] | 121.901 | 117.949 | 85.944 | 424.826 | 8.203 |
| 100k | MultithreadedScalar[2] | 55.423 | 54.758 | 41.544 | 243.478 | 18.043 |
| 100k | MultithreadedScalar[4] | 29.328 | 29.079 | 24.419 | 204.691 | 34.097 |
| 100k | MultithreadedScalar[8] | 17.362 | 17.239 | 13.603 | 54.863 | 57.597 |
| 100k | MultithreadedScalar[16] | 12.117 | 11.949 | 9.796 | 30.571 | 82.528 |
| 100k | MultithreadedScalar[32] | 12.869 | 13.045 | 10.334 | 18.595 | 77.708 |
| 100k | MultithreadedAVX2[1] | 71.272 | 68.230 | 31.155 | 238.222 | 14.031 |
| 100k | MultithreadedAVX2[2] | 34.588 | 34.553 | 24.567 | 51.346 | 28.912 |
| 100k | MultithreadedAVX2[4] | 21.166 | 19.398 | 16.562 | 233.679 | 47.245 |
| 100k | MultithreadedAVX2[8] | 13.636 | 13.351 | 10.864 | 31.717 | 73.337 |
| 100k | MultithreadedAVX2[16] | 9.345 | 9.267 | 7.802 | 12.994 | 107.014 |
| 100k | MultithreadedAVX2[32] | 9.897 | 9.655 | 8.501 | 35.654 | 101.036 |
| 100k | CUDA-Persistent | 5.179 | 4.109 | 3.092 | 32.864 | 193.100 |
| 100k | CUDA-BlockParallel | 5.977 | 3.874 | 3.228 | 49.523 | 167.314 |
| 1m | Scalar | 1201.549 | 1188.660 | 1083.945 | 1457.894 | 0.832 |
| 1m | AVX2 | 704.070 | 702.794 | 638.119 | 753.075 | 1.420 |
| 1m | MultithreadedScalar[16] | 104.739 | 105.498 | 89.111 | 120.498 | 9.548 |
| 1m | MultithreadedScalar[32] | 109.420 | 113.318 | 94.367 | 120.688 | 9.139 |
| 1m | MultithreadedAVX2[16] | 75.824 | 76.599 | 68.807 | 87.073 | 13.188 |
| 1m | MultithreadedAVX2[32] | 84.239 | 83.928 | 76.665 | 90.986 | 11.871 |
| 1m | CUDA-Persistent | 62.595 | 42.754 | 31.836 | 285.107 | 15.976 |
| 1m | CUDA-BlockParallel | 60.596 | 50.496 | 38.911 | 225.284 | 16.503 |

## Correctness and validation

Pre-change CUDA Release built and passed 288 tests. Updated CUDA Release passes all 309; CPU-only Release passes 243 and skips 66 hardware tests, with zero failures. Tests compare scalar, existing naive CUDA and block-parallel CUDA, including dimension tails/small dimensions, zero magnitudes, K edge cases, ties, finite extremes/cancellation, invalid inputs, index updates/reuse and concurrent first queries. The naive kernel and 17 CPU source/header files are unchanged: [preservation evidence](block-preservation.txt).

Every warm-up and measured result matched scalar IDs/ranks and scores within 1e-6. Exact ID checksums match across all backends in each dataset run. Summation order differs, so score bitwise equality is not an API guarantee. Actual checksums and maximum score errors:

| Vectors | Backend | Score checksum | ID checksum | Max score error |
|---|---|---:|---:|---:|
| 1k | Scalar | 287.063861355 | 11688052443322439949 | 0 |
| 1k | CUDA-Persistent | 287.063861355 | 11688052443322439949 | 0 |
| 1k | CUDA-BlockParallel | 287.063861355 | 11688052443322439949 | 0 |
| 10k | Scalar | 360.302542560 | 6942403294749163753 | 0 |
| 10k | CUDA-Persistent | 360.302542560 | 6942403294749163753 | 0 |
| 10k | CUDA-BlockParallel | 360.302542560 | 6942403294749163753 | 0 |
| 100k | Scalar | 425.896235436 | 4652264718457849833 | 0 |
| 100k | CUDA-Persistent | 425.896235436 | 4652264718457849833 | 0 |
| 100k | CUDA-BlockParallel | 425.896235436 | 4652264718457849833 | 0 |
| 1m | Scalar | 48.008113995 | 10511493593234606401 | 0 |
| 1m | CUDA-Persistent | 48.008113995 | 10511493593234606401 | 0 |
| 1m | CUDA-BlockParallel | 48.008113995 | 10511493593234606401 | 0 |

## Reproduction and raw logs

Build and test commands are in [README](../../README.md#build-and-test). Benchmarks ran sequentially after compilation/tests:

```powershell
./out/build-cuda/vectorpulse_benchmark.exe --vectors 1000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 1,2,4,8,16,32 --cuda-paths persistent
./out/build-cuda/vectorpulse_benchmark.exe --vectors 10000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 1,2,4,8,16,32 --cuda-paths persistent
./out/build-cuda/vectorpulse_benchmark.exe --vectors 100000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 1,2,4,8,16,32 --cuda-paths persistent
./out/build-cuda/vectorpulse_benchmark.exe --vectors 1000000 --dimension 768 --queries 10 --top-k 10 --warmups 1 --repetitions 3 --threads 16,32 --cuda-paths persistent
```

Raw logs: [1K](block-1k.txt), [10K](block-10k.txt), [100K](block-100k.txt), [1M](block-1m.txt). Each retains all measured rounds, phase distributions, build costs and checksums. [Pre-change tests](block-baseline-tests.txt), [CUDA tests](block-cuda-tests.txt), [CPU-only tests](block-cpu-tests.txt).

## Changed files

- `include/vectorpulse/cuda_search_backend.h`: explicit kernel selection; naive remains the default.
- `src/cuda_search_backend.cpp`, `src/cuda_runtime_bridge.h`, `src/cuda_runtime_stub.cpp`: kernel dispatch through existing persistent storage and CPU-only compatibility.
- `src/cuda_runtime.cu`: new block-per-vector shared-memory kernel, grid bounds and dispatch; original naive kernel preserved.
- `tests/cuda_search_backend_test.cpp`: three-mode comparisons and block-specific correctness/lifecycle tests.
- `benchmarks/scalar_benchmark.cpp`: paired block/naive persistent variants and optional historical transfer baseline.
- `README.md`, `docs/architecture.md`, `docs/benchmarks/block-*`: architecture, validation and measured evidence.

No CMake, VectorStore or CPU backend changes were made in this milestone. Existing uncommitted changes from earlier milestones remain.

## Interpretation and limitations

Observed mean end-to-end speedups over naive persistent CUDA were 1.06x, 1.17x, 0.87x and 1.03x at 1K, 10K, 100K and 1M respectively. At 100K, block-parallel was 15.4% slower by mean end-to-end latency. At 1M, the 3.3% mean improvement is small relative to timing variation: block median end-to-end was 50.496 ms versus naive 42.754 ms, and median kernel time was 17.969 ms versus 13.893 ms. These results do not establish a reliable 1M speedup. No block-size tuning or further kernel optimization was performed to change the outcome.

CPU was faster than both GPU variants at 1K and 10K. Block-parallel beat the best measured CPU mean at 100K and 1M by 1.56x and 1.25x, respectively; naive persistent CUDA also beat CPU at those sizes. These comparisons use the same current benchmark runs, not historical CPU measurements.

Kernel and end-to-end speedups are reported separately: smaller kernel time cannot remove query allocation/transfer/synchronization, all-score download or serial CPU Top-K. Shared-memory reduction changes floating-point addition order, and arbitrarily close scores on other inputs can reorder despite passing tolerance and exact-rank tests here. Exact equal scores still use ascending ID.

No affinity, fixed clocks, exclusive GPU access, hardware-counter profiling or timing-outlier removal was used. WDDM/display and other machine activity affect both CUDA events and host timing; small-workload end-to-end differences should not be overgeneralized. Inputs/sample counts differ across dataset sizes, and 1M tests only two CPU thread counts. Historical milestone results are not used as the paired baseline.

Persistent index update/ownership limits are unchanged: writes need external exclusion; rebuilds retain old and new GPU allocations until publication; allocation/copy failures were not fault-injected. Query scratch is still allocated per call. The block size is fixed at 256 and has not been tuned. No next-stage CUDA optimization is included.
