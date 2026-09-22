# VectorPulse

VectorPulse is a C++20 in-memory vector search engine. CUDA now keeps the vector
database resident on the GPU and offers FP64/FP32 naive and block-parallel kernels,
alongside the four unchanged CPU backends.
VectorStore and SearchBackend APIs remain unchanged; scalar is still the default.

## Backends and behavior

- ScalarSearchBackend
- AVX2SearchBackend (single-threaded AVX2/FMA)
- MultithreadedScalarSearchBackend
- MultithreadedAVX2SearchBackend
- CudaSearchBackend (naive or block-parallel CUDA, persistent storage, CPU Top-K)

Vectors have fixed dimensions and unique string IDs. Searches exhaustively score
all vectors and order results by descending cosine score, then ascending ID for
exact ties. Zero-magnitude vectors score zero; K=0 returns nothing; oversized K
returns all entries. Invalid dimensions/duplicate IDs throw std::invalid_argument;
missing IDs throw std::out_of_range. CUDA additionally rejects NaN/infinity inputs
and checks returned scores are finite; existing CPU behavior is unchanged.

No GPU Top-K, warp-level optimization, CUDA graphs, multi-GPU support,
ANN indexes, quantization, networking, disk persistence, Docker, Python bindings, or
frontend is implemented. No FAISS, cuBLAS, Thrust, or search library implements
the similarity computation.

## Build and test

Requires CMake 3.20+, a C++20 compiler with std::jthread, and GoogleTest 1.17.0
(downloaded on first configuration unless supplied through FetchContent).
CUDA is OFF by default; this build requires neither CUDA headers nor a toolkit:

```powershell
cmake -S . -B build -DVECTORPULSE_ENABLE_CUDA=OFF
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA requires a CUDA compiler supporting C++20, a compatible NVIDIA GPU, and a
suitable driver. From a Visual Studio developer terminal with Ninja and nvcc:

```powershell
cmake -S . -B out/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DVECTORPULSE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build out/build-cuda
ctest --test-dir out/build-cuda --output-on-failure
```

89 targets the measured RTX 4080 SUPER; set architectures appropriate for your
device. If not specified, the project defaults to 75 (real code plus PTX).
CMake checks for a CUDA compiler only when enabled. If none is found, it warns
and builds the CPU-only stub. A found but unusable compiler/toolkit remains a
configuration error. Configure output and the benchmark explicitly report whether
CUDA was compiled. Only .cu files receive CUDA flags; CPU AVX2 flags remain
confined to the original AVX2 kernel. CUDA uses a static runtime.

This machine had a GPU driver but no CUDA Toolkit. Validation used NVIDIA's
verified CUDA 12.9 Update 1 redistributable compiler/runtime/header archives,
assembled locally under ignored out/cuda-12.9. No GPU driver was replaced.
For that local setup, also pass:

```powershell
-DCUDAToolkit_ROOT="E:/Project VectorPulse/out/cuda-12.9" -DCMAKE_CUDA_COMPILER="E:/Project VectorPulse/out/cuda-12.9/bin/nvcc.exe"
```

The measured nvcc version is 12.9.86 (cudart package 12.9.79), using MSVC
19.44.35228.0 as host compiler. The device image is sm_89 with compute_89 PTX;
--fmad=false disables FMA contraction, with no fast math. The block reduction
changes summation order; correctness uses floating-point tolerance.
Normal CPU flags remain /O2 /Ob2 /DNDEBUG /MD /std:c++20 /W4 /permissive-;
the AVX2 kernel alone adds /arch:AVX2.

VECTORPULSE_ENABLE_AVX2=OFF still disables SIMD; tests and benchmarks can be
omitted with VECTORPULSE_BUILD_TESTS=OFF / VECTORPULSE_BUILD_BENCHMARKS=OFF.
For single-config generators, use -DCMAKE_BUILD_TYPE=Release and omit --config/-C.

## Use CUDA

```cpp
#include "vectorpulse/cuda_search_backend.h"
#include "vectorpulse/vector_store.h"
#include <memory>

const auto info = vectorpulse::CudaSearchBackend::device_info();
if (info.available) {
    vectorpulse::VectorStore store{768,
        std::make_unique<vectorpulse::CudaSearchBackend>(768)};
    // store.add(), store.get(), store.search() use the existing interface.
}
// info.compiled distinguishes a CPU-only binary; info.reason explains unavailability.
```

is_compiled() and is_supported() are also available. This backend uses logical
CUDA device 0. Construction checks runtime/device/kernel-image availability and
throws a useful runtime_error when unavailable. It does not silently fall back.
An unavailable CUDA backend does not stop CPU benchmarks or CPU tests.

## Persistent CUDA database

```text
Build/update -> flatten host entries -> upload immutable GPU matrix (once)
Query -> query H2D -> selected CUDA cosine kernel -> scores D2H -> CPU Top-K
```

`CudaSearchBackend(dimension)` defaults to `CudaStorageMode::Persistent`.
Host entries remain available through `get()`. The first nonempty, nonzero-K
search builds the GPU index; subsequent searches reuse it. `build_index()` can
perform this work eagerly and returns flatten/upload/total timings. A successful
`add()` makes the append-only index stale; the next build/search coalesces pending
inserts into one full rebuild. Rejected inserts leave the cached index valid.

A replacement is published only after allocation and upload succeed. Failed
rebuilds throw and are retried; searches never use an outdated matrix. RAII owns
the GPU memory, with shared immutable snapshots and a mutex protecting cache
construction/publication. Callers must exclude insertion from searches, eager
builds, retrieval and other insertion. Concurrent read-only searches have
independent query/score buffers. Default-stream/device synchronization can still
serialize GPU work. Transactional rebuilding temporarily needs both old and new
GPU matrices plus host flattening scratch.

For the historical transfer baseline, construct
`CudaSearchBackend(dimension, CudaStorageMode::PerQuery)`. This mode flattens,
allocates and uploads the full database for every query. Both modes use the same
unchanged naive kernel: 256 threads per block, one thread per vector, serial
double-precision accumulation over dimensions, FP32 scores. All scores return
to the CPU for deterministic descending-score/ascending-ID Top-K.

Persistent queries still allocate query/score buffers and events. No GPU Top-K,
pinned memory, transfer overlap or norm caching is added.
CUDA calls are checked; per-query normal-path cleanup errors throw, and RAII
index destruction/unwinding cleanup reports errors without throwing. Logical
device 0 owns allocations, and the caller's device selection is restored.

## Block-parallel CUDA

Select the new kernel explicitly; the default remains the naive persistent baseline:

```cpp
vectorpulse::CudaSearchBackend block_backend{
    768, vectorpulse::CudaStorageMode::Persistent,
    vectorpulse::CudaKernel::BlockParallel};
```

One 256-thread block processes one vector. Thread t visits dimensions t, t+256,
t+512, etc. Two shared-memory arrays hold double dot-product and norm partials
(4096 bytes per block). A binary tree reduction uses `__syncthreads()` at every
stage; thread 0 writes the FP32 score. Query norm is computed once on the CPU.
Dimension tails and dimensions smaller than a block use zero partials for idle
threads. No warp-level primitives or GPU Top-K are used.

The kernel reuses the same persistent index, RAII, rebuild rules, query transfers,
score download and CPU Top-K implementation. Each backend instance owns its own
index. Block-parallel with PerQuery storage is rejected. The original naive
kernel is unchanged and remains selectable for direct comparison.

## FP32 accumulation variants

`CudaKernel::Naive` and `CudaKernel::BlockParallel` remain the unchanged FP64
baselines and the default stays naive FP64. Select `CudaKernel::NaiveFP32` or
`CudaKernel::BlockParallelFP32` with persistent storage to use float products,
dot-product/vector-norm accumulation and (for block mode) float shared reduction.
Block size, reduction tree, barriers, transfers and CPU Top-K are unchanged.
Query norm is still computed once in double on the CPU. Final normalization uses
double to avoid overflow in the product of norms; these are FP32 **accumulation**
variants, not kernels with zero FP64 instructions. FMA contraction remains disabled.

The FP32 validation tolerance is absolute score error **1e-5**; FP64 remains 1e-6.
All tests and benchmark queries require exact Top-K IDs and rankings as well as
that tolerance. A mismatch fails validation. This is measured correctness on the
recorded inputs, not a mathematical guarantee that float accumulation preserves
arbitrarily close rankings for every input/dimension. There is no hidden CPU
rescoring, epsilon tie sorting or FP64 fallback. Exact computed-score ties still
use ascending IDs.

FP32 input safety: every nonzero component of stored vectors and queries must
satisfy `sqrt(FLT_MIN) <= abs(value) <= sqrt(FLT_MAX / (4 * dimension))`;
zeros are allowed. Here FLT_MIN means the smallest positive **normal** float.
This conservative bound avoids product underflow and reserves accumulator
headroom. Inputs outside the bound throw `invalid_argument` with guidance to
rescale or select FP64; FP64's existing finite-input range remains unchanged.
The bound protects representability, not a universal error/ranking bound.

See [FP32 measurements and Nsight profiles](docs/benchmarks/fp32-results.md).

## Benchmark methodology

```powershell
./out/build-cuda/vectorpulse_benchmark.exe --vectors 100000 --dimension 768 --queries 100 --top-k 10 --warmups 1 --repetitions 3 --threads 1,2,4,8,16,32 --cuda-paths persistent
```

The existing framework uses identical seed-42 data/queries, an untimed scalar
reference, full-query warm-up rounds and configurable measured repetitions.
Backend order is shuffled with seed 2026 between measured rounds. --threads
selects both threaded CPU families. All numeric options are positive. Defaults
remain 10,000 vectors, dimension 128, 100 queries, K=10, one warm-up, five repeats,
threads 1,2,4,8,16. CUDA variants run when available and report device/runtime details.
`--cuda-paths persistent` compares all four persistent FP64/FP32 variants;
`--cuda-paths all` (default) additionally retains the old full-transfer FP64 baseline.
`--cpu-paths scalar` retains the scalar reference/benchmark while omitting other
CPU variants, bounding memory for four GPU matrices at 1M vectors. Default is `all`.
The persistent index is explicitly built before warm-up; its one-time cost is
reported separately. Every persistent query verifies zero database H2D bytes.

Every result from warm-up and measurement is compared with scalar for exact IDs
and ranks plus score tolerance 1e-6 (FP64) or 1e-5 (FP32). All measured scores and IDs feed checksums;
a mismatch returns failure. SIMD/GPU rounding can reorder arbitrarily close
scores on other inputs; exact scalar ranking for every possible input is not
claimed. Deterministic exact-score ties always use IDs.

Mean/median/min/max summarize measured individual queries; each round mean and
its distribution are also printed. QPS=1000/mean milliseconds. Scalar speedup and
CPU/GPU comparisons use end-to-end means from the same run. Data generation,
insertion, warm-ups, checksums, validation and returned-result destruction are
outside timing. Query execution, result retention and backend scratch cleanup
are inside timing. Queries run sequentially, not as concurrent clients.

CUDA additionally reports distributions for:

1. Query norm, plus database flattening only in PerQuery mode (steady_clock).
2. Query-path H2D, including completion synchronization (steady_clock):
   full matrix plus query in PerQuery mode, query only in persistent mode.
3. Kernel-only interval (CUDA events).
4. Score D2H, complete before return (steady_clock).
5. CPU result materialization, validation and Top-K (steady_clock).
6. Profiled end-to-end, including allocations, event overhead and cleanup.

The outer benchmark end-to-end latency determines CUDA QPS and speedup. Kernel
latency is never substituted for end-to-end. Phases need not sum to total:
allocation, event management, device checks, synchronization bookkeeping and
cleanup are intentionally accounted for only in total. search_profiled() exposes
per-call timings, lazy index-build timings, and transfer byte counts without
mutable last-query state; ordinary search() uses the same
implementation. Transfer timing follows NVIDIA's documented pageable-copy
[completion behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

All compared stores coexist. The FP32 comparison uses `--cpu-paths scalar`:
five host stores (15.36 GB raw vectors at 1M) and four persistent GPU matrices
(12.288 GB total at 1M), plus metadata, queries/results and temporary build scratch.
GPU memory capacity must accommodate all selected variants. Clocks, CPU affinity
and exclusive GPU access are not controlled; display/system activity adds noise.

## Measured results and validation

See [FP32 accumulation results](docs/benchmarks/fp32-results.md) for the latest
paired numerical/performance comparison. See [block-parallel CUDA results](docs/benchmarks/block-results.md) for paired
1K, 10K, 100K and 1M measurements at 768 dimensions: kernel-only and end-to-end
latency, measured speedups, best CPU variants, distributions and raw logs.
Historical [persistent-storage results](docs/benchmarks/persistent-results.md)
and [naive transfer baseline](docs/benchmarks/milestone3-results.md) remain available.

Before the FP32 change, CUDA Release built and passed all 309 tests. Updated CUDA
Release passes all 347 tests; CPU-only Release passes 243 and skips 104 hardware
tests, with zero failures. Parameterized comparisons test scalar, both FP64/FP32 kernels and
the historical PerQuery baseline across dimensions 3/7/128/384/768/769 and counts 1/257/513.
Additional block tests cover dimensions 1/255/256/257/767/769/1025, empty/K edge
cases, ties, finite extremes, cancellation, invalid inputs, index reuse/rebuilds,
query-only transfers and concurrent first queries.
See [CUDA tests](docs/benchmarks/fp32-cuda-tests.txt) and
[CPU-only tests](docs/benchmarks/fp32-cpu-tests.txt).

The naive kernel, CPU backends and VectorStore are unchanged. CPU Top-K and full
score transfers remain O(n). Shared-memory reduction changes summation order;
all measured IDs/ranks must match scalar, with absolute score tolerance 1e-6
for FP64 and 1e-5 for FP32. Exact
score ties retain ascending-ID ordering, but arbitrarily close scores can reorder
on other inputs. Rebuild failures are handled transactionally but have not been
fault-injected. Only the recorded Windows/MSVC/CUDA machine was measured.
See [architecture](docs/architecture.md) for implementation and ownership details.

## Nsight profiling

The earlier [FP64 Nsight Compute report](docs/benchmarks/nsight-results.md) compares both unchanged kernels at
10K, 100K and 1M vectors (768 dimensions). It records duration, occupancy, SM/DRAM
throughput, load/warp/branch efficiency, registers, shared memory and launch shape.
Counter evidence shows small-grid underutilization at 10K, and FP64/reduction
overhead at larger sizes. A profiling-only driver and collection script were added;
no backend or kernel algorithm changed. Native reports and raw counters are retained.
