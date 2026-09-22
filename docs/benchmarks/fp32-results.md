# FP32 CUDA accumulation: correctness, benchmarks and Nsight

Added explicit `CudaKernel::NaiveFP32` and `CudaKernel::BlockParallelFP32` variants. Both FP64 definitions remain unchanged, and naive FP64 remains the default. The FP32 variants use the existing persistent GPU index, identical launch/reduction structure, transfers and CPU Top-K. Dot product, vector norm and block shared reduction accumulate in float. Query norm is still computed once on the CPU in double, and final GPU normalization stays double to prevent norm-product overflow. `--fmad=false` remains unchanged: FMA fusion is not another optimization in this comparison.

## Numerical contract and validation

FP32 validation requires **exact Top-K IDs/ranks** against ScalarSearchBackend and **absolute score error <= 1e-5** (`CudaSearchBackend::fp32_score_tolerance`). FP64 keeps 1e-6. All tested/benchmarked cases pass these checks; any mismatch fails the executable. There is no epsilon-based tie sorting, hidden scalar reranking or fallback to FP64. Equal computed scores retain ascending-ID ordering. Different rounded scores can reorder arbitrarily close candidates on other inputs; these measurements do not prove rank/error bounds for arbitrary datasets and dimensions.

Each nonzero FP32 input component must satisfy `sqrt(FLT_MIN) <= abs(x) <= sqrt(FLT_MAX/(4*dimension))`; FLT_MIN is the smallest positive normal float, and zeros are valid. This conservative validation keeps products normal and gives dot/norm accumulation headroom. Invalid magnitudes, NaN and infinity throw before mutation/search; callers can rescale or choose FP64. The guard is a representability bound, not a universal numerical-error guarantee. FP64 still accepts its existing full finite input range.

CUDA Release: **347 tests passed**. CPU-only Release: **243 passed, 104 hardware skips**, zero failures. Tests compare all five supported storage/kernel combinations, full results and Top-K, dimension tails, empty/K edge cases, exact ties, cancellation, safe small/large magnitudes, rejected ranges and persistent updates. [CUDA tests](fp32-cuda-tests.txt), [CPU tests](fp32-cpu-tests.txt).

## Benchmark method

Hardware: i9-14900F (24 cores, hardware_concurrency 32), RTX 4080 SUPER (16 GB, 80 SMs, compute 8.9), driver 610.60, Windows WDDM/shared display GPU. CUDA 12.9 / nvcc 12.9.86, MSVC 19.44, Release sm_89/compute_89, static CUDA runtime. No clock/affinity controls or exclusive GPU access.

All runs use 768 dimensions, K=10, seed-42 identical data/query sets and an untimed scalar reference. One full warm-up round precedes three measured repetitions; backend order is shuffled with seed 2026. 10K/100K use 100 queries (300 measured); 1M uses 10 queries (30 measured). All four GPU indices are built before warm-up, so reported latencies are steady-state and exclude one-time builds. Each query verifies zero database upload and exactly 3072 query bytes.

`--cpu-paths scalar` retains the scalar reference and comparison while bounding memory: five host stores total 15.36 GB raw values at 1M; four GPU matrices total 12.288 GB, plus metadata/scratch. Kernel latency uses CUDA events; end-to-end wraps query execution/result retention including query validation, norm, allocations, H2D, kernel synchronization, all-score D2H, CPU Top-K and cleanup. Correctness/checksums are outside timing. No samples are discarded.

## Paired benchmark means

Latency units are ms/query. Error is the maximum absolute difference over returned Top-K scores in warm-up and measured queries. Full-result unit tests provide additional score coverage.

| Vectors | Variant | Kernel ms | End-to-end ms | Max absolute score error |
|---|---|---:|---:|---:|
| 10k | CUDA-Persistent | 0.287295 | 1.196 | 0 |
| 10k | CUDA-BlockParallel | 0.206243 | 1.194 | 0 |
| 10k | CUDA-NaiveFP32 | 0.099319 | 1.054 | 2.08616257e-07 |
| 10k | CUDA-BlockParallelFP32 | 0.055711 | 0.586 | 2.23517418e-08 |
| 100k | CUDA-Persistent | 2.407305 | 5.377 | 0 |
| 100k | CUDA-BlockParallel | 1.756359 | 4.069 | 0 |
| 100k | CUDA-NaiveFP32 | 2.583606 | 5.292 | 2.38418579e-07 |
| 100k | CUDA-BlockParallelFP32 | 0.495563 | 3.009 | 2.98023224e-08 |
| 1m | CUDA-Persistent | 20.984103 | 51.821 | 0 |
| 1m | CUDA-BlockParallel | 18.401401 | 46.265 | 0 |
| 1m | CUDA-NaiveFP32 | 31.319831 | 57.255 | 2.08616257e-07 |
| 1m | CUDA-BlockParallelFP32 | 4.875732 | 32.407 | 2.98023224e-08 |

FP64 names: `CUDA-Persistent` = naive FP64; `CUDA-BlockParallel` = block FP64.

| Vectors | Shape | FP64/FP32 kernel speedup | FP64/FP32 end-to-end speedup |
|---|---|---:|---:|
| 10k | Naive | 2.893x | 1.135x |
| 10k | Block | 3.702x | 2.038x |
| 100k | Naive | 0.932x | 1.016x |
| 100k | Block | 3.544x | 1.352x |
| 1m | Naive | 0.670x | 0.905x |
| 1m | Block | 3.774x | 1.428x |

## Nsight Compute FP32/FP64 comparison

Nsight Compute 2025.2.1.0, same toolchain and unchanged clock-control/cache protocol as [earlier profiling](nsight-results.md). The [FP32 collection script](run-nsight-fp32.ps1) runs all four kernels at each size as an administrator for counter access, without changing the global counter policy. Each isolated case uses one seed-42 query, three warm-ups, and three captured launches. Replay flushes caches (`--cache-control all`) and leaves clocks unmanaged (`--clock-control none`).

Profiler duration is `gpu__time_duration.sum`, not the instrumented driver's event/total times, which include replay overhead. FP32 utilization is `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` (the FP32 arithmetic pipeline including ordinary FADD/FMUL; this does not mean FMA fusion was enabled). FP64 utilization is `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active`. These percentages have different hardware peak denominators and are not fractions that should sum to 100.

| Vectors | Kernel | Nsight duration ms | FP32 pipeline % active | FP64 pipeline % active | Occupancy % | DRAM % peak | Registers/thread | Static shared B/block |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | naive | 0.274965 | 0.12 | 84.90 | 16.45 | 16.27 | 38 | 0 |
| 10,000 | block | 0.183083 | 1.37 | 84.88 | 78.44 | 24.69 | 25 | 4096 |
| 10,000 | naive32 | 0.149440 | 1.82 | 0.70 | 15.65 | 28.63 | 27 | 0 |
| 10,000 | block32 | 0.052768 | 8.75 | 44.32 | 80.16 | 83.69 | 25 | 2048 |
| 100,000 | naive | 1.442027 | 0.11 | 80.64 | 68.35 | 30.14 | 38 | 0 |
| 100,000 | block | 1.661248 | 1.38 | 85.24 | 78.90 | 26.12 | 25 | 4096 |
| 100,000 | naive32 | 1.373931 | 1.02 | 0.40 | 65.25 | 31.68 | 27 | 0 |
| 100,000 | block32 | 0.470784 | 8.89 | 45.00 | 81.61 | 91.59 | 25 | 2048 |
| 1,000,000 | naive | 14.050240 | 0.11 | 77.53 | 95.47 | 31.47 | 38 | 0 |
| 1,000,000 | block | 16.670816 | 1.38 | 85.27 | 80.33 | 26.33 | 25 | 4096 |
| 1,000,000 | naive32 | 13.984469 | 0.91 | 0.35 | 95.54 | 31.76 | 27 | 0 |
| 1,000,000 | block32 | 4.773408 | 8.75 | 44.30 | 80.82 | 92.79 | 25 | 2048 |

`naive`/`block` are FP64, `naive32`/`block32` are FP32 accumulation. Residual FP64 is expected because normalization remains double. Block shared arrays shrink from 4096 to 2048 bytes as a direct consequence of changing element type; no reduction topology or synchronization optimization was added.

## End-to-end distributions and checksums

Each latency cell is mean / median / min / max (ms); exact ID checksums match across all five variants within each size. Score checksums can differ under the documented tolerance.

| Vectors | Variant | End-to-end | Score checksum | ID checksum |
|---|---|---|---:|---:|
| 10k | Scalar | 8.773 / 6.159 / 5.703 / 29.107 | 360.302542560 | 6942403294749163753 |
| 10k | CUDA-Persistent | 1.196 / 0.758 / 0.647 / 10.777 | 360.302542560 | 6942403294749163753 |
| 10k | CUDA-BlockParallel | 1.194 / 0.740 / 0.539 / 10.335 | 360.302542560 | 6942403294749163753 |
| 10k | CUDA-NaiveFP32 | 1.054 / 0.611 / 0.431 / 7.290 | 360.302550428 | 6942403294749163753 |
| 10k | CUDA-BlockParallelFP32 | 0.586 / 0.548 / 0.415 / 2.417 | 360.302542783 | 6942403294749163753 |
| 100k | Scalar | 118.652 / 115.673 / 78.717 / 335.401 | 425.896235436 | 4652264718457849833 |
| 100k | CUDA-Persistent | 5.377 / 3.526 / 2.957 / 33.931 | 425.896235436 | 4652264718457849833 |
| 100k | CUDA-BlockParallel | 4.069 / 3.600 / 3.255 / 9.802 | 425.896235436 | 4652264718457849833 |
| 100k | CUDA-NaiveFP32 | 5.292 / 3.356 / 2.904 / 31.701 | 425.896234632 | 4652264718457849833 |
| 100k | CUDA-BlockParallelFP32 | 3.009 / 2.433 / 2.003 / 11.537 | 425.896234721 | 4652264718457849833 |
| 1m | Scalar | 1142.884 / 1166.264 / 596.969 / 1374.787 | 48.008113995 | 10511493593234606401 |
| 1m | CUDA-Persistent | 51.821 / 42.488 / 31.759 / 300.534 | 48.008113995 | 10511493593234606401 |
| 1m | CUDA-BlockParallel | 46.265 / 44.552 / 33.626 / 66.031 | 48.008113995 | 10511493593234606401 |
| 1m | CUDA-NaiveFP32 | 57.255 / 38.864 / 30.026 / 339.741 | 48.008112699 | 10511493593234606401 |
| 1m | CUDA-BlockParallelFP32 | 32.407 / 30.978 / 19.976 / 55.109 | 48.008114174 | 10511493593234606401 |

## Reproduction and artifacts

```powershell
./out/build-cuda/vectorpulse_benchmark.exe --vectors 10000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 16 --cuda-paths persistent --cpu-paths scalar
./out/build-cuda/vectorpulse_benchmark.exe --vectors 100000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 16 --cuda-paths persistent --cpu-paths scalar
./out/build-cuda/vectorpulse_benchmark.exe --vectors 1000000 --dimension 768 --queries 10 --top-k 10 --warmups 1 --repetitions 3 --threads 16 --cuda-paths persistent --cpu-paths scalar
# Administrator PowerShell for NVIDIA performance counters:
./docs/benchmarks/run-nsight-fp32.ps1
```

Benchmark logs: [10K](fp32-10k.txt), [100K](fp32-100k.txt), [1M](fp32-1m.txt). Machine-readable [summary](fp32-results.json) retains distributions and per-launch Nsight counters. [Baseline preservation](fp32-preservation.json).

| Profile | Native report | Metrics | Details | Collection |
|---|---|---|---|
| 10000-naive | [ncu-rep](fp32-nsight/10000-naive.ncu-rep) | [CSV](fp32-nsight/10000-naive-metrics.csv) | [details](fp32-nsight/10000-naive-details.txt) | [log](fp32-nsight/10000-naive-collection.txt) |
| 10000-block | [ncu-rep](fp32-nsight/10000-block.ncu-rep) | [CSV](fp32-nsight/10000-block-metrics.csv) | [details](fp32-nsight/10000-block-details.txt) | [log](fp32-nsight/10000-block-collection.txt) |
| 10000-naive32 | [ncu-rep](fp32-nsight/10000-naive32.ncu-rep) | [CSV](fp32-nsight/10000-naive32-metrics.csv) | [details](fp32-nsight/10000-naive32-details.txt) | [log](fp32-nsight/10000-naive32-collection.txt) |
| 10000-block32 | [ncu-rep](fp32-nsight/10000-block32.ncu-rep) | [CSV](fp32-nsight/10000-block32-metrics.csv) | [details](fp32-nsight/10000-block32-details.txt) | [log](fp32-nsight/10000-block32-collection.txt) |
| 100000-naive | [ncu-rep](fp32-nsight/100000-naive.ncu-rep) | [CSV](fp32-nsight/100000-naive-metrics.csv) | [details](fp32-nsight/100000-naive-details.txt) | [log](fp32-nsight/100000-naive-collection.txt) |
| 100000-block | [ncu-rep](fp32-nsight/100000-block.ncu-rep) | [CSV](fp32-nsight/100000-block-metrics.csv) | [details](fp32-nsight/100000-block-details.txt) | [log](fp32-nsight/100000-block-collection.txt) |
| 100000-naive32 | [ncu-rep](fp32-nsight/100000-naive32.ncu-rep) | [CSV](fp32-nsight/100000-naive32-metrics.csv) | [details](fp32-nsight/100000-naive32-details.txt) | [log](fp32-nsight/100000-naive32-collection.txt) |
| 100000-block32 | [ncu-rep](fp32-nsight/100000-block32.ncu-rep) | [CSV](fp32-nsight/100000-block32-metrics.csv) | [details](fp32-nsight/100000-block32-details.txt) | [log](fp32-nsight/100000-block32-collection.txt) |
| 1000000-naive | [ncu-rep](fp32-nsight/1000000-naive.ncu-rep) | [CSV](fp32-nsight/1000000-naive-metrics.csv) | [details](fp32-nsight/1000000-naive-details.txt) | [log](fp32-nsight/1000000-naive-collection.txt) |
| 1000000-block | [ncu-rep](fp32-nsight/1000000-block.ncu-rep) | [CSV](fp32-nsight/1000000-block-metrics.csv) | [details](fp32-nsight/1000000-block-details.txt) | [log](fp32-nsight/1000000-block-collection.txt) |
| 1000000-naive32 | [ncu-rep](fp32-nsight/1000000-naive32.ncu-rep) | [CSV](fp32-nsight/1000000-naive32-metrics.csv) | [details](fp32-nsight/1000000-naive32-details.txt) | [log](fp32-nsight/1000000-naive32-collection.txt) |
| 1000000-block32 | [ncu-rep](fp32-nsight/1000000-block32.ncu-rep) | [CSV](fp32-nsight/1000000-block32-metrics.csv) | [details](fp32-nsight/1000000-block32-details.txt) | [log](fp32-nsight/1000000-block32-collection.txt) |

## Interpretation

Block FP32 shows a consistent kernel benefit: unprofiled CUDA-event mean speedups are 3.70x, 3.54x and 3.77x at 10K, 100K and 1M. End-to-end gains are smaller (2.04x, 1.35x, 1.43x), since host Top-K, transfers, allocations and synchronization remain. At 1M, block FP32's CPU Top-K phase alone averaged 24.645 ms of its 32.407 ms end-to-end mean.

Nsight corroborates the block improvement: at 100K, duration falls from 1.661248 to 0.470784 ms (3.53x). FP64 pipeline utilization falls from 85.24% to 45.00%, FP32 pipeline utilization rises from 1.38% to 8.89%, and DRAM throughput rises from 26.12% to 91.59% of peak. At 1M, block FP32 reaches 92.79% DRAM throughput. This supports a shift toward a bandwidth limit after reducing accumulation cost. Residual double normalization is visible in the FP64 counters and is explicitly retained.

Naive FP32 is not a reliable large-dataset win. Nsight shows FP64 utilization dropping from 80.64% to 0.40% at 100K, but duration only improves from 1.442027 to 1.373931 ms; at 1M it is nearly unchanged (14.050240 to 13.984469 ms). Removing FP64 accumulation does not remove the existing memory-access/latency limitations of the naive layout. In unprofiled measurements the 1M FP32 mean end-to-end is worse (57.255 vs 51.821 ms), despite a better median (38.864 vs 42.488 ms). Large kernel/end-to-end outliers remain in both variants; no samples were removed and no general naive-FP32 speedup is claimed.

Maximum observed benchmark score errors were 2.384185791e-7 for naive FP32 and 2.980232239e-8 for block FP32. All tested and benchmarked IDs/ranks matched scalar exactly. These observations support the documented 1e-5 validation tolerance for these workloads, not a universal bound for arbitrary inputs.

## Scope and limits

Changed: CUDA public kernel enum/tolerance, host FP32 range validation and dispatch, the two added kernel definitions, tests, benchmark/profile drivers, new profiling script and performance docs. The original FP64 kernels and all 17 CPU implementation/header files remain unchanged. Persistent storage, query transfers, block size, shared tree/barriers, CPU Top-K and compiler arithmetic flags are retained.

No warp optimization, GPU Top-K, cached vector norms or further features were added. Exact ranking is established for the tested inputs, not arbitrary near-ties. Input magnitude checks are conservative and reject some values that FP32 could handle in particular datasets. FP32 norm/dot cancellation can still accumulate rounding error. Timing uses a shared WDDM machine; small mean differences can be noise, and replay measurements do not establish application end-to-end speedup. The 1M benchmark keeps four matrices resident, while Nsight profiles one backend per process; those are different memory/cache conditions.
