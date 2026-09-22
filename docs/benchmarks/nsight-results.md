# Nsight Compute profile: naive vs block-parallel CUDA

Both existing production kernels were profiled without changing their algorithms, flags, storage, transfers or CPU Top-K. Only a small profiling driver and its CMake target were added. All backend/kernel source and header hashes remain unchanged.

## Collection and reproducibility

- GPU: NVIDIA GeForce RTX 4080 SUPER, AD103, compute 8.9, **80 SMs**, 16 GB; driver 610.60, WDDM/shared display GPU. CPU: i9-14900F, 24 cores / 32 logical threads.
- Nsight Compute CLI **2025.2.1.0**, build 35987062, NVIDIA archive 2025.2.1.3. CUDA 12.9 / nvcc 12.9.86, MSVC 19.44, Release, sm_89/compute_89, `--fmad=false`. No new compilation flags or line-info instrumentation.
- NVIDIA archive SHA256: `e9d558654c98d83049969d133b98922b53ab8f4e3ba9e0a37bdb5e2ff300b7de`, verified against the [CUDA 12.9.1 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json). Extracted under ignored `out/nsight`; no global toolkit/driver install.
- Each case uses 768 dimensions, K=10, seed 42, one deterministic query, an eager persistent index build, three warm-up launches, then five driver iterations. Nsight captures **three launches after the three warm-ups**, with **34 replay passes per captured launch**. All eight driver results are compared to scalar.
- `--cache-control all` flushes caches for replay; `--clock-control none` leaves clocks unmanaged. These are counter-profiling measurements, not a repeat of the earlier sequential-query end-to-end benchmark. No other applications were closed, no clocks or driver security settings were changed.
- Counter access initially failed with `ERR_NVGPUCTRPERM`. With user approval, one administrator PowerShell job collected all six cases. The global counter-access policy was not changed. See [NVIDIA counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

Reproduce from the project root after the existing Release CUDA build:

```powershell
./out/build-cuda/vectorpulse_cuda_profile.exe 100000 naive
./out/build-cuda/vectorpulse_cuda_profile.exe 100000 block
# Run in an administrator PowerShell terminal when GPU counters require elevation:
./docs/benchmarks/run-nsight.ps1
```

The [collection script](run-nsight.ps1) selects LaunchStats, Occupancy, SpeedOfLight, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis_Tables, SchedulerStats, WarpStateStats and SourceCounters. It profiles 10K/100K/1M for each kernel and exports native reports, raw metrics and interpreted details. See the [Nsight CLI manual](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

**Timing boundary:** tables below use Nsight `gpu__time_duration.sum`, converted from ns to ms, averaged over three captured launches. The driver's own event/total timings during replay contain instrumentation/replay overhead and must not be used as application latency. Native reports and per-launch metrics are retained; no outliers were removed.

## Kernel duration and utilization

| Vectors | Kernel | Duration mean ms | Min–max ms | Achieved occupancy % | Compute/SM % peak | DRAM GB/s read+write | DRAM % peak |
|---:|---|---:|---|---:|---:|---:|---:|
| 10,000 | naive | 0.275083 | 0.274944–0.275360 | 16.45 | 41.42 | 116.29 | 16.17 |
| 10,000 | block | 0.183061 | 0.182912–0.183136 | 78.30 | 83.94 | 169.45 | 23.56 |
| 100,000 | naive | 1.443243 | 1.436704–1.447264 | 65.98 | 78.00 | 217.92 | 30.29 |
| 100,000 | block | 1.713045 | 1.670304–1.796608 | 78.64 | 82.60 | 183.69 | 25.53 |
| 1,000,000 | naive | 14.640864 | 14.600224–14.664160 | 95.30 | 74.99 | 216.86 | 31.31 |
| 1,000,000 | block | 17.372203 | 17.363840–17.380832 | 80.30 | 84.28 | 181.80 | 26.06 |

Theoretical occupancy is 100% for both kernels at all sizes. Achieved occupancy measures active warps against the SM limit; Compute/SM is peak-normalized throughput, not the `nvidia-smi` busy percentage. DRAM GB/s is the sum of measured read/write bandwidth in decimal units.

| Vectors | Naive/block kernel speedup | Block duration change |
|---:|---:|---:|
| 10,000 | 1.503x | -33.45% |
| 100,000 | 0.843x | +18.69% |
| 1,000,000 | 0.843x | +18.66% |

## Memory and execution efficiency

| Vectors | Kernel | Global-load bytes/32B sector | Sector utilization % | Sectors/load request | Active-lane efficiency % | Predicated-on lane efficiency % | Branch efficiency % |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10,000 | naive | 4.00 | 12.50 | 16.474 | 99.84 | 99.78 | 100.00 |
| 10,000 | block | 32.00 | 100.00 | 4.000 | 94.47 | 89.44 | 97.21 |
| 100,000 | naive | 4.00 | 12.50 | 16.438 | 100.00 | 99.94 | 100.00 |
| 100,000 | block | 32.00 | 100.00 | 4.000 | 94.47 | 89.44 | 97.21 |
| 1,000,000 | naive | 4.00 | 12.50 | 16.497 | 100.00 | 99.94 | 100.00 |
| 1,000,000 | block | 32.00 | 100.00 | 4.000 | 94.47 | 89.44 | 97.21 |

Global load efficiency is reported explicitly as **measured sector byte utilization**, derived from Nsight's bytes-per-sector metric / 32, plus sectors/request. This avoids claiming an unavailable legacy `gld_efficiency` metric. It aggregates vector and query loads, including broadcasts; it is not DRAM bandwidth utilization or an eightfold estimate of avoidable DRAM traffic. Active/predicated-on lane efficiency divides the corresponding measured average threads per warp instruction by 32. Branch efficiency is Nsight's uniform branch-target percentage; predicated-off lanes are reported separately.

## Launch configuration and resource usage

| Vectors | Kernel | Grid blocks x threads/block | Waves/SM | Registers/thread used (allocated) | Static / dynamic shared B/block | Driver shared B/block |
|---:|---|---|---:|---|---|---:|
| 10,000 | naive | 40 x 256 | 0.08 | 38 (40) | 0 / 0 | 1024 |
| 10,000 | block | 10,000 x 256 | 20.83 | 25 (32) | 4096 / 0 | 1024 |
| 100,000 | naive | 391 x 256 | 0.81 | 38 (40) | 0 / 0 | 1024 |
| 100,000 | block | 100,000 x 256 | 208.33 | 25 (32) | 4096 / 0 | 1024 |
| 1,000,000 | naive | 3,907 x 256 | 8.14 | 38 (40) | 0 / 0 | 1024 |
| 1,000,000 | block | 1,000,000 x 256 | 2083.33 | 25 (32) | 4096 / 0 | 1024 |

Naive uses zero algorithmic shared memory, plus 1024 bytes of driver-reserved shared memory per block. BlockParallel uses 4096 static bytes (two 256-double arrays), zero dynamic bytes, plus 1024 driver bytes: 5120 total. Register allocation is rounded above the compiler-reported count. Both kernels support six resident blocks/SM at this block size; the warp limit is six blocks, register limits are six (naive) and eight (block), shared-memory limits are sixteen and twelve respectively. Thus shared-memory capacity does **not** explain an occupancy collapse; none is measured at 100K. Shared-memory bank conflict count was zero in all captured samples.

## Bottleneck evidence

| Vectors | Kernel | FP64 pipeline % active | Barrier stall cycles/issued inst | Barrier share of warp latency % | Predicated-on thread instructions (billions) | L1 global-load sector hit % |
|---:|---|---:|---:|---:|---:|---:|
| 10,000 | naive | 84.89 | 0.00 | 0.00 | 0.083412 | 87.81 |
| 10,000 | block | 84.88 | 40.81 | 42.41 | 0.409890 | 49.01 |
| 100,000 | naive | 81.73 | 0.00 | 0.00 | 0.834101 | 80.32 |
| 100,000 | block | 85.24 | 41.91 | 43.47 | 4.098900 | 49.18 |
| 1,000,000 | naive | 77.10 | 0.00 | 0.00 | 8.341002 | 74.43 |
| 1,000,000 | block | 85.27 | 42.53 | 43.22 | 40.989000 | 49.19 |

Barrier share is the ratio of two warp-scheduler latency counters, **not** a fraction of wall time or a promised speedup. Stalls overlap across warps. The instruction total is the predicated-on SASS thread-instruction count, not a FLOP count. Nsight identifies FP64 as the highest-utilized compute pipeline in both kernels.

## Why the result changes with size

**10K: the naive grid is too small to fill this GPU.** Forty blocks cannot occupy all 80 SMs, and each active SM initially has just one 8-warp block against a 48-warp limit. Measured achieved occupancy is 16.45% and SM throughput 41.42%. BlockParallel launches 10,000 blocks, reaches 78.30% achieved occupancy and 83.94% SM throughput, and improves load coalescing from about 16.5 to 4 sectors/request. This supports the 1.503x profiled speedup (0.275083 → 0.183061 ms). The smaller-workload benefit is supported by measured grid/occupancy/utilization evidence.

**100K: naive already supplies substantial parallelism, while reduction overhead remains.** Its 391 blocks yield 65.98% occupancy and 78.00% SM throughput. BlockParallel raises occupancy to 78.64%, but duration rises from 1.443243 to 1.713045 ms (+18.69%). Both heavily use FP64 (81.73%/85.24% of the active-cycle peak), while DRAM is only 30.29%/25.53% of peak. Better sector utilization therefore does not remove the principal measured compute pressure.

The block kernel executes 4.0989 billion predicated-on thread instructions at 100K versus 0.8341 billion for naive (about 4.91x). Its source-level reduction adds two trees of 255 additions per vector and nine full-block barriers. Measured barrier stalls are 41.91 cycles per issued instruction, about 43.47% of its warp latency counter, versus zero for naive. Active-lane/predication and branch efficiency also decrease during reduction and the single-lane final result. These counters support extra reduction/synchronization and instruction overhead as the cost that offsets better coalescing once the naive grid is sufficiently large. They do not separately assign an exact number of milliseconds to each cause.

Naive's strided accesses are partially served by cache reuse: at 100K its L1 global-load sector hit rate is about 81%, versus about 49% for block. Both read roughly the matrix size from DRAM despite very different L1 sector counts. Low naive sector utilization should not be interpreted as eight times the DRAM traffic. Neither shared-memory bank conflicts nor shared-memory capacity is implicated by these captures.

**1M: more naive work removes the small-grid advantage further.** Naive reaches 95.30% achieved occupancy. BlockParallel remains at 80.30% and retains barrier/instruction overhead. Profiled means are 14.640864 ms naive and 17.372203 ms block (+18.66%). This does not reproduce the prior benchmark's small block mean advantage. That benchmark already reported worse block medians (kernel 17.969 vs 13.893 ms, end-to-end 50.496 vs 42.754 ms) and large outliers. The profiler is consistent with those medians. The previous small mean advantage is not evidence of a robust 1M kernel win.

These are evidence-based bottleneck inferences, not a controlled attribution of every cycle. Counter collection uses replay/cache flushing, a fixed query and unmanaged clocks; the earlier benchmark used many queries without replay. No unprofiled end-to-end speedup is inferred from the profiler duration. No optimization was implemented in response.

## Validation, artifacts and limits

CUDA Release: all 309 tests passed. CPU-only Release: 243 passed, 66 hardware skips, zero failures. Every profiling case matched scalar IDs/ranks with maximum measured score error zero. Persistent queries transferred only 3072 query bytes and performed no index rebuild/database upload. See [CUDA tests](nsight/cuda-tests.txt), [CPU tests](nsight/cpu-tests.txt), [source preservation](nsight/source-preservation.json) and [numeric summary](nsight/summary.json).

| Case | Native report | Raw metrics | Details | Collection log |
|---|---|---|---|---|
| 10000-naive | [ncu-rep](nsight/10000-naive.ncu-rep) | [CSV](nsight/10000-naive-metrics.csv) | [details](nsight/10000-naive-details.txt) | [log](nsight/10000-naive-collection.txt) |
| 10000-block | [ncu-rep](nsight/10000-block.ncu-rep) | [CSV](nsight/10000-block-metrics.csv) | [details](nsight/10000-block-details.txt) | [log](nsight/10000-block-collection.txt) |
| 100000-naive | [ncu-rep](nsight/100000-naive.ncu-rep) | [CSV](nsight/100000-naive-metrics.csv) | [details](nsight/100000-naive-details.txt) | [log](nsight/100000-naive-collection.txt) |
| 100000-block | [ncu-rep](nsight/100000-block.ncu-rep) | [CSV](nsight/100000-block-metrics.csv) | [details](nsight/100000-block-details.txt) | [log](nsight/100000-block-collection.txt) |
| 1000000-naive | [ncu-rep](nsight/1000000-naive.ncu-rep) | [CSV](nsight/1000000-naive-metrics.csv) | [details](nsight/1000000-naive-details.txt) | [log](nsight/1000000-naive-collection.txt) |
| 1000000-block | [ncu-rep](nsight/1000000-block.ncu-rep) | [CSV](nsight/1000000-block-metrics.csv) | [details](nsight/1000000-block-details.txt) | [log](nsight/1000000-block-collection.txt) |

Each case is three samples in one profiler invocation. Clock frequency and display/background activity were not controlled; replay and software counters change execution conditions. Higher occupancy/SM utilization is not by itself proof of faster execution. No Nsight Systems timeline was collected; CPU/WDDM scheduling attribution is therefore not proven. No counter failure or missing requested category remains after elevation.

For metric semantics, consult NVIDIA's [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/), particularly occupancy, replay, warp state statistics and L1/TEX sectors per request. Definitions in this report map directly to these raw metric names:

| Quantity | Raw metric / calculation |
|---|---|
| Kernel duration | `gpu__time_duration.sum` / 1e6 → ms |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| SM utilization | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM GB/s | (`dram__bytes_read.sum.per_second` + `dram__bytes_write.sum.per_second`) / 1e9 |
| DRAM utilization | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| Global-load sector utilization | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` / 32 × 100 |
| Global sectors/request | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` |
| Active / predicated-on lane efficiency | `smsp__thread_inst_executed_per_inst_executed.ratio` / 32 × 100; analogous `smsp__thread_inst_executed_pred_on_per_inst_executed.ratio` |
| Branch efficiency | `smsp__sass_average_branch_targets_threads_uniform.pct` |
| Registers | `launch__registers_per_thread`, `launch__registers_per_thread_allocated` |
| Shared memory | `launch__shared_mem_per_block_static`, `_dynamic`, `_driver` |
| Launch | `launch__grid_size`, `launch__block_size`, `launch__waves_per_multiprocessor` |
| FP64 utilization | `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active` |
| Barrier / total warp latency | `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` / `smsp__average_warp_latency_per_inst_issued.ratio` |
| Predicated-on thread instructions | `thread_inst_executed_true` |
