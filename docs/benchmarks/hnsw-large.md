# Large-scale HNSW and exact search benchmark

The shared benchmark supports a named million-vector preset:

```powershell
./out/build-cuda/vectorpulse_hnsw_benchmark.exe --preset large --threads 16
```

This selects N=1,000,000, dimension=768, K=10 and efSearch=40,80,160,320.
Defaults remain M=16, efConstruction=200, 100 held-out queries, one warm-up
round, three measured rounds, and seed=42. Numeric CLI options override the
preset regardless of their order. Use --help for the complete option list.
The preset is opt-in; ordinary CTest runs use a tiny dataset.

Build with VECTORPULSE_BUILD_BENCHMARKS=ON (default). CUDA is optional:
a CPU-only build still measures scalar, supported AVX2 backends and HNSW.
An unavailable AVX2 or CUDA backend is explicitly reported as SKIP on stderr.
Errors in an available backend fail the benchmark rather than silently falling back.

## Compared implementations

- ExactScalar: default VectorIndex and existing scalar backend.
- ExactAVX2: existing AVX2/FMA backend.
- ExactMultithreadedAVX2: existing threaded AVX2 backend, configurable --threads
  (default 16; includes the calling thread).
- ExactCUDA_BlockParallelFP64: existing persistent block-parallel FP64 CUDA
  kernel, when available. No FP32 accumulation or per-query database upload.
- HNSW: one CPU graph, queried at each requested efSearch without rebuilding.

All data and held-out queries are generated once using the same mt19937_64 stream
and uniform FP32 components in [-1,1]. Every backend receives identical vector
bits, string IDs, insertion order and queries. Scalar ground-truth Top-K results
are retained after the scalar index is destroyed.

## Timing and memory

Backends are built, warmed, measured and destroyed sequentially in the order above.
Only the generated dataset and one index coexist, bounding host/GPU memory at
large N. This changes the earlier benchmark's simultaneous scalar/HNSW lifetimes.
Exact backend order is fixed; HNSW efSearch order is shuffled between measured
rounds (seed 2026). Clock drift, CPU affinity and background activity are not
controlled, so these are local measurements, not universal speedup claims.

Build time includes index construction, ID/vector copies and insertion. CUDA
build also includes the eager persistent matrix flatten/upload via build_index().
Data generation, device availability probing and scalar ground truth are excluded.
HNSW progress is printed every 10,000 insertions; that small logging cost is
included in build time. HNSW's repeated build number is one shared graph build.

Query latency covers ordinary search(), including returned-result construction,
backend allocations and scratch cleanup. It excludes result destruction,
validation, memory sampling and recall calculations. CUDA latency is end-to-end,
not kernel-only. Reported QPS is 1000 / mean query milliseconds. The CSV also
contains median and nearest-rank p95 over individual measured calls.

Host memory is sampled using Windows GetProcessMemoryInfo or Linux /proc/self/status;
unsupported platforms report -1. Fields give whole-process resident bytes before
build, after build and after queries, plus process-lifetime peak resident bytes.
These include the generated data, metadata, allocator retention, driver/runtime
memory and retained ground truth. They are not isolated index sizes; differences
between samples are not reliable allocator accounting. The lifetime peak can
include earlier backends' temporary buffers. No background memory sampler runs
inside query timing.

The raw FP32 dataset payload alone is N*D*4 = 3,072,000,000 bytes
(2.861 GiB). Each index owns another host copy plus IDs/metadata, and HNSW adds
graph adjacency. CUDA reports actual persistent database bytes returned by
build_index(), excluding per-query buffers and CUDA context/runtime overhead.
The benchmark keeps only one CUDA index alive.

## Correctness and recall

Recall@K is mean overlap with scalar Top-K divided by min(K,N); at this preset
it is Recall@10. Each result is checked for unique/known IDs, result count,
finite cosine score and descending-score/ascending-ID ordering.

HNSW/scalar scores must equal the existing scalar helper exactly.
AVX2 and FP64 CUDA scores use the existing 1e-6 absolute tolerance.
SIMD/GPU summation can reorder very close scores, so recall is measured for those
backends too rather than hardcoded to 100%. Their outputs are not reranked.
Full-breadth HNSW additionally must reproduce exact scalar IDs/order/scores.

All search implementations and CUDA kernels are unchanged.


## Recorded 1M x 768 run

Windows 11 x64, Intel Core i9-14900F (32 logical processors), 64 GiB RAM,
NVIDIA RTX 4080 SUPER (16 GiB), MSVC 19.44 Release, CUDA 12.9, driver API
version 13030. Command: `--preset large --threads 16`, using all other defaults.

| Backend | Build s | Mean ms | QPS | Recall@10 | Process RSS after build GiB | Process RSS after queries GiB | GPU database GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ExactScalar | 1.437 | 952.092 | 1.05 | 100.0% | 6.033 | 9.759 | 0.000 |
| ExactAVX2 | 1.911 | 536.076 | 1.87 | 100.0% | 9.762 | 9.762 | 0.000 |
| ExactMultithreadedAVX2 | 1.828 | 65.432 | 15.28 | 100.0% | 9.763 | 9.765 | 0.000 |
| ExactCUDA_BlockParallelFP64 | 5.758 | 40.448 | 24.72 | 100.0% | 9.874 | 9.874 | 2.861 |
| HNSW ef=40 | 13593.278 | 2.578 | 387.83 | 0.6% | 10.201 | 10.202 | 0.000 |
| HNSW ef=80 | 13593.278 | 4.690 | 213.23 | 1.0% | 10.201 | 10.202 | 0.000 |
| HNSW ef=160 | 13593.278 | 7.725 | 129.45 | 1.8% | 10.201 | 10.202 | 0.000 |
| HNSW ef=320 | 13593.278 | 15.004 | 66.65 | 3.5% | 10.201 | 10.202 | 0.000 |

HNSW construction took **3 hours 46 minutes 33 seconds**. All four rows use
that same graph. At the requested query breadths it is faster, but **Recall@10
is only 0.6% to 3.5%**. This initial graph/configuration does not provide useful
high-recall search on this million-vector uniform random, 768-dimensional
dataset. These measurements do not establish behavior on structured embeddings.
No search algorithm was tuned or changed to improve these numbers.

All four exact backends measured 100% Recall@10 on the same held-out queries.
The fastest measured exact backend was persistent CUDA block FP64 at 40.448 ms
(24.72 QPS); multithreaded AVX2 with 16 threads averaged 65.432 ms (15.28 QPS).

RAM figures are whole-process snapshots and include retained allocator pages
from earlier backends. The higher AVX2/HNSW after-build RSS does not establish
an equivalent increase in isolated index storage. The process-lifetime peak was
12.735 GiB, reached during the CUDA phase and retained in the later peak counter.
HNSW's after-build process RSS was 10.201 GiB. GPU database storage was 2.861 GiB.

[Complete raw output](hnsw-1m-768.txt) includes unrounded timings, p95, memory
snapshots and checksums. [HNSW construction progress](hnsw-1m-768-build.txt)
records all 100 insertion milestones. All measurements passed score, count,
uniqueness and ordering validation; no results were reranked or substituted.

## Validation

CUDA Release: all 400 CTest cases passed. CPU-only Release: 286 passed and
114 CUDA cases skipped, with zero failures. This includes exact search, HNSW,
persistence, the installed CMake consumer, Python with/without NumPy, the small
large-preset override smoke test and invalid-preset/thread-count checks.
