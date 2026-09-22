# Architecture

## Backends

```text
VectorStore owns SearchBackend
                 +-- ScalarSearchBackend
                 +-- AVX2SearchBackend
                 +-- MultithreadedScalarSearchBackend
                 +-- MultithreadedAVX2SearchBackend
                 +-- CudaSearchBackend
                       +-- CudaKernel::Naive (default)
                       +-- CudaKernel::BlockParallel (persistent only)
```

SearchBackend and VectorStore remain unchanged. The CUDA host store now caches
an immutable GPU database through its runtime bridge. No CPU source/header changes.
The CPU implementations retain their scalar/SIMD arithmetic and established
balanced contiguous partitioning, caller participation, local Top-K and join/merge
behavior. Existing SIMD capability detection and per-source compile flags remain.

## CUDA components and optional build

- include/vectorpulse/cuda_search_backend.h: public backend, availability/device
  information, and per-call profiling result types. No CUDA headers are exposed.
- src/cuda_search_backend.cpp: host entry/ID storage, finite-input validation,
  transactional index caching, flattening/query norm, results and CPU Top-K.
- src/cuda_runtime_bridge.h: opaque index ownership and score-computation boundary.
- src/cuda_runtime.cu: device detection, RAII buffers/events, transfers, naive
  and block-parallel similarity kernels, and GPU/transfer timing.
- src/cuda_error.h: checked CUDA status handling and nonthrowing cleanup reporting.
- src/cuda_runtime_stub.cpp: CPU-only availability and defensive index/scoring stubs.

VECTORPULSE_ENABLE_CUDA defaults OFF. Only an enabled, discovered CUDA compiler
causes enable_language(CUDA), toolkit discovery and .cu compilation. Otherwise
the stub is linked, and availability explains that CUDA was not compiled. CMake
and benchmark output explicitly report the compiled state. CPU warning/AVX2 flags
are scoped to C++ or the original SIMD source; CUDA has separate C++20 compiler
settings, a static cudart dependency and --fmad=false, without fast math.

Default CUDA architectures are 75 real/PTX unless overridden. The measured build
uses 89 real/PTX. CUDA device 0 must have a compatible compiled kernel image;
cudaFuncGetAttributes checks resolution before construction succeeds. A driver
error, no visible device, prohibited compute mode, or incompatible image makes
availability false with an explanation. Construction then throws runtime_error.
There is no auto fallback and no multi-device scheduling. CPU backends remain
usable when CUDA is unavailable. The caller's current CUDA device is restored.

## Index lifetime and query flow

`CudaSearchBackend` defaults to persistent storage. Its append-only host entries
retain vectors and IDs for `get()` and CPU result materialization. A private,
opaque `detail::CudaIndex` owns the contiguous device-0 matrix through RAII;
CUDA types never appear in public headers. The CPU-only bridge supplies throwing
stubs and availability reporting, preserving toolkit-free builds.

```text
add() succeeds -> host entry count differs from indexed_count
build_index() or next nonempty/nonzero-K search:
  lock cache mutex -> check count -> flatten rows -> allocate replacement
  -> full H2D + synchronize -> publish immutable snapshot -> unlock
steady-state search:
  validate query -> compute norm -> acquire shared snapshot
  -> allocate private query/score scratch -> query-only H2D
  -> selected cosine kernel -> all scores D2H -> CPU Top-K -> free scratch
```

Insertion validates dimensions, finiteness and unique IDs and rolls back on map
allocation failure. Because the API is append-only and `get()` returns const
references, entry count identifies the host generation without an overflow-prone
revision counter. Invalid inserts do not invalidate the index. Multiple successful
inserts before a search cause one rebuild. Empty stores and K=0 skip GPU work
after query validation. Matrix byte multiplication and device grid limits are
checked before the corresponding allocations/launches.

`acquire_index()` serializes cache inspection, construction and publication with
one mutex. The lock is released before query execution; a shared immutable
snapshot keeps the matrix alive for each reader. Queries use independent query,
score, event, result and timing state. Simultaneous first searches publish one
matrix. As with the existing backends, callers must exclude insertion from all
other operations, and destruction from active operations. This does not provide
concurrent writers or mutation through retained vector references.

Allocation/upload errors propagate before cache publication. The old snapshot
is retained, the entry counts still differ, and subsequent searches retry rather
than return stale results. Successful replacement releases the old snapshot when
its final owner leaves. This requires peak VRAM for old plus new matrices during
updates; there is no incremental upload or in-place capacity management.
Flattening scratch exists only during builds. `build_index()` exposes eager build
timings; a valid index or PerQuery mode returns a no-op report.

## Kernel and Top-K

The naive kernel is unchanged. Row i contains entry i's FP32 values in row-major
order. A 256-thread block assigns one thread to each row; grid size is ceil(n/256).
Each thread serially accumulates double dot product and vector squared norm,
using the CPU-computed double query norm. Zero magnitude yields zero; other
scores are cast to float. `--fmad=false` and existing architecture flags remain.
The naive implementation remains byte-for-byte unchanged (normalized line endings).

`CudaKernel::BlockParallel` selects the new kernel in the same backend, only with
persistent storage. One block processes one row; grid size is n, checked against
the device limit before launch. Threads visit dimensions in strides of 256. At
768 dimensions each of the 256 threads contributes three products and squared
values. Adjacent threads read adjacent dimensions in the unchanged row-major
matrix. Two 256-element double shared arrays (4096 bytes) hold the partial sums.
A binary reduction combines strides 128,64,...,1, with a full-block barrier after
initial writes and every reduction stage (nine barriers). Thread 0 computes the
final score using the once-computed CPU query norm. The block size is fixed and
asserted to be a power of two; it is not benchmark-tuned.

Dimensions below 256 and dimension tails leave unused lanes with zero partials;
all lanes still reach every barrier. Zero query norm and out-of-range row returns
are block-uniform and occur before barriers. Zero vector norm yields zero.
Double accumulation preserves finite FP32 extremes but the reduction changes
addition order relative to scalar/naive. No warp shuffle, warp-specialized tail,
transposition, vector norm cache or GPU Top-K is added.

Kernel selection changes only dispatch within the persistent query path. Both
kernels use identical index ownership, query scratch, upload/download, event
measurement and CPU result processing. Each benchmark backend instance owns a
separate resident matrix; no storage implementation or VectorStore logic is
copied into a new backend class. The default constructor retains the naive
persistent kernel. PerQuery plus BlockParallel throws invalid_argument.

Every score returns to the host. The existing result materialization checks
finiteness and maps rows to IDs. `partial_sort` handles a subset and `sort` handles
all results, ordered by descending score then ascending ID. Exact tested ranks
and scores within 1e-6 match scalar. Arbitrarily close scores on other arithmetic
paths can still reorder; tests do not claim universal bitwise equivalence.

## Baseline and ownership

`CudaStorageMode::PerQuery` with the naive kernel preserves the old full-transfer route for paired
benchmarks. It allocates a fresh matrix, query and score buffer, flattens/uploads
all host vectors each search, and runs exactly the same kernel and CPU Top-K.
Persistent mode owns the matrix across calls and allocates only query/score/event
scratch per search. Neither mode uses pinned buffers, custom streams, overlap,
GPU Top-K, or buffer pools.

RAII cleans up initialized buffers/events after allocation, copy or launch errors.
Normal per-query cleanup checks and propagates CUDA errors; destructors log
cleanup failures without throwing. Index destruction selects the owning device
and restores the caller's selection. CUDA diagnostics include failed operation,
error name/message and source location. Default-stream event synchronization
waits for kernel completion; H2D uses device synchronization. These semantics
can serialize concurrent GPU work or include unrelated device work. No device
reset occurs. Resource-failure paths have not been fault-injected.

## Measurement boundaries

The benchmark builds the persistent matrix explicitly before warm-up and reports
its one-time flatten, H2D and total cost separately. Query end-to-end timing
includes validation, query norm, snapshot acquisition, allocations, transfers,
kernel synchronization, host result materialization/Top-K and scratch cleanup.
A lazy rebuild is included in search total and separately exposed as
`timings.index_build`; steady-state benchmark queries assert no rebuild or
database upload. H2D byte counters distinguish index-build bytes, per-query
database bytes and query bytes (3072 at dimension 768).

Host steady_clock measures flatten/norm, synchronized H2D, synchronous D2H and
CPU Top-K. CUDA events measure only the selected kernel. Allocation, event overhead,
device selection, synchronization bookkeeping and cleanup are included in total
but not assigned to individual phases, so phase sums need not equal total.
Index-build total also includes temporary host matrix destruction and replacement
cleanup. Benchmarked end-to-end wraps the full call and result retention;
checksums, correctness comparisons and returned-result destruction are outside.

All variants receive identical seed-42 data and queries; scalar produces untimed
reference results. Full warm-up rounds precede repeated measurements, with
seed-2026 shuffled backend order per repetition. All IDs/ranks and scores are
validated outside timing; checksums consume all measured results. Reports retain
mean/median/min/max, QPS, phase distributions and scalar speedups. CPU comparisons
are paired within each dataset run, and the best result is only the best among
measured variants. No affinity, clock controls or exclusive GPU access is used.

## FP64 block milestone validation history

Before the block milestone, CUDA Release passed all 288 tests. That milestone passed 309;
CPU-only Release passes 243 with 66 hardware skips. The three supported storage/
kernel combinations are parameterized against scalar and naive CUDA over
3/7/128/384/768/769 dimensions and 1/257/513 vectors. Block-specific tests include
1/255/256/257/767/769/1025 dimensions, tails, empty and K edge cases, exact ties,
finite extremes/cancellation, rejected inputs, repeated index rebuild/reuse and
concurrent first readers. The naive kernel text and all 17 pre-existing CPU
implementation/header hashes are verified unchanged.

`--cuda-paths persistent` benchmarks naive persistent and block-parallel CUDA
against CPU variants; `all` additionally includes the original PerQuery baseline.
At all four requested sizes, identical seeded inputs, warm-ups and repeated
measurements check exact IDs/ranks and 1e-6 score tolerance against scalar.
Kernel-only and end-to-end means, distributions and paired CPU comparisons are
reported in [block-parallel results](benchmarks/block-results.md). Hardware noise
and changed summation order remain limitations; no further CUDA optimization is
part of this milestone.

## Nsight profiling

The [measured Nsight Compute report](benchmarks/nsight-results.md) compares both unchanged kernels at
10K, 100K and 1M vectors (768 dimensions). It records duration, occupancy, SM/DRAM
throughput, load/warp/branch efficiency, registers, shared memory and launch shape.
Counter evidence shows small-grid underutilization at 10K, and FP64/reduction
overhead at larger sizes. A profiling-only driver and collection script were added;
no backend or kernel algorithm changed. Native reports and raw counters are retained.

## FP32 accumulation milestone

The persistent query dispatcher now also selects `NaiveFP32` and
`BlockParallelFP32`. Their launch shape and shared reduction topology match the
FP64 definitions, which remain unchanged. Only per-dimension products and dot/
norm accumulation (including block shared arrays) switch from double to float.
The block shared arrays shrink from 4096 to 2048 bytes. No warp primitive,
GPU Top-K, norm cache, transfer optimization or CPU backend redesign was added.

Query norm stays double and is computed once on the host; final per-vector
normalization stays double to keep norm products representable. Consequently
Nsight may report residual FP64 utilization for FP32 variants. `--fmad=false`
is unchanged; FP32 comparisons do not include a separate FMA optimization.

Both variants use the existing index, RAII, rebuilds, query upload, score download
and host Top-K path. Only the FP64 naive kernel supports historical PerQuery
storage. FP32 validates finite values and conservatively bounds each nonzero
component to [sqrt(FLT_MIN), sqrt(FLT_MAX/(4*dimension))] to avoid product
underflow and norm overflow. Zero components remain valid. Rejected inputs do
not mutate the store/index. FP64 retains the wider finite-input contract.

FP32 tests/benchmarks require exact IDs and rankings against scalar with absolute
score error <= 1e-5; FP64 remains <= 1e-6. Equal computed scores retain ID ordering.
Near-ties on arbitrary inputs may reorder under float rounding; input magnitude
bounds do not prove rank equivalence or that tolerance for arbitrary dimensions.
There is no fallback/rescoring that would hide the numerical or performance cost.
Tests include full-result comparisons, tails, cancellation, safe magnitude
extremes, invalid ranges, ties and persistent updates. CUDA: 347 passed; CPU-only:
243 passed, 104 hardware skips. See [FP32 results](benchmarks/fp32-results.md) for
actual error, paired kernel/end-to-end timing and FP32/FP64 Nsight counters.
