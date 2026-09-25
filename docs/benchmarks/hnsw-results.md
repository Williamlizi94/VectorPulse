# Initial CPU HNSW measurements

This historical run predates the exact-backend comparison extension. See the
[1M x 768 comparison](hnsw-large.md) for the expanded benchmark, its sequential
backend lifetimes, and current results.

Measured on Windows x64, Intel Core i9-14900F (32 logical processors), MSVC
19.44 Release (/O2), using the CPU-only build. Queries are single-threaded.
The existing scalar VectorIndex is the ground truth and timed exact baseline;
these numbers are not comparisons against AVX2 or CUDA.

Dataset: 10,000 vectors, dimension 64, 100 held-out queries, K=10, uniform
FP32 components in [-1,1], seed 42. HNSW: M=16, efConstruction=200, seed 42.
One graph is reused for all six efSearch values. Each configuration receives
one warm-up round and three measured rounds (300 queries).

| Index | efSearch | Build ms | Mean query ms | p95 ms | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact scalar | exhaustive | 1.878 | 0.795 | 1.042 | 1,257 | 100.0% |
| HNSW | 10 | 6,727.540 | 0.086 | 0.140 | 11,575 | 36.1% |
| HNSW | 20 | 6,727.540 | 0.125 | 0.161 | 7,969 | 51.1% |
| HNSW | 40 | 6,727.540 | 0.257 | 0.365 | 3,890 | 70.1% |
| HNSW | 80 | 6,727.540 | 0.399 | 0.572 | 2,503 | 87.3% |
| HNSW | 160 | 6,727.540 | 0.745 | 1.106 | 1,342 | 96.9% |
| HNSW | 320 | 6,727.540 | 1.311 | 1.729 | 763 | 99.8% |

The repeated HNSW build time represents one build, not six independent builds.
At efSearch=80 this run achieves approximately 2x scalar throughput with 87.3%
recall. Near-perfect recall at efSearch=320 is slower than scalar on this dataset.
The initial implementation favors correctness and uses the existing cosine helper
for each evaluated candidate; it does not cache norms or use SIMD-specific graph
scoring. Synthetic uniform data and these sizes do not establish production
embedding performance. CPU affinity, clocks and background activity were not
controlled; timings are one local run, not a cross-machine guarantee.

## Reproduce

```powershell
cmake -S . -B build -DVECTORPULSE_ENABLE_CUDA=OFF
cmake --build build --config Release
./build/Release/vectorpulse_hnsw_benchmark.exe --vectors 10000 --dimension 64 --queries 100 --top-k 10 --m 16 --ef-construction 200 --ef-search 10,20,40,80,160,320 --warmups 1 --repetitions 3 --seed 42
```

See [raw output](hnsw-10k-64.txt) for unrounded means, medians, p95s and
checksums, and [the API/design guide](../hnsw.md) for timing/recall definitions.
Change M/efConstruction in separate runs to measure construction trade-offs.
The CTest smoke case also includes efSearch=N and requires identical exact
Top-K IDs, scores and order.

## Compatibility validation

- CUDA Release: all 398 CTest cases pass, including persistence, CPU/CUDA exact
  search, the installed CMake consumer, and Python tests with/without NumPy.
- CPU-only Release: 284 cases pass, 114 CUDA cases skip, zero failures.
- The installed shared-library CMake consumer passes exact persistence and HNSW search.
- The Python source distribution includes the new HNSW header and implementation.
- Sixteen HNSW correctness cases and one benchmark smoke case are included.
- The existing exact benchmark passes a small run across CPU and all CUDA paths.
- A clean virtual environment installs the Python package from the working tree
  and passes import/add/search/save/load without NumPy.

No exact-search implementation or CUDA kernel was changed. HNSW is CPU-only
and has no persistence or Python API in this change.
