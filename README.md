# VectorPulse

VectorPulse is a C++20 vector search engine. Milestone 2A adds a single-threaded
AVX2/FMA CPU backend alongside the unchanged Milestone 1 scalar baseline.

## Features and behavior

- In-memory vectors with unique string IDs and fixed dimensionality
- Manual cosine similarity and brute-force Top-K search
- Interchangeable `ScalarSearchBackend` and `AVX2SearchBackend`
- Deterministic ranking: descending score, then ascending ID for exact ties
- GoogleTest correctness tests and a deterministic backend comparison benchmark

`VectorStore` defaults to scalar. Duplicate IDs and invalid dimensions throw
`std::invalid_argument`; missing IDs throw `std::out_of_range`. Zero-magnitude
vectors have similarity `0.0`. Zero K returns no results; oversized K returns
all entries. No CUDA, multithreading, networking, persistence, HNSW, Docker,
Python bindings, or frontend is implemented.

## Build and test

Requires CMake 3.20+, a C++20 compiler, and internet access on first configuration
for GoogleTest. From a Developer PowerShell on Windows:

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

For a single-config generator such as Ninja:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests and benchmarks can be disabled with `VECTORPULSE_BUILD_TESTS=OFF` and
`VECTORPULSE_BUILD_BENCHMARKS=OFF`. To build without the SIMD kernel:

```powershell
cmake -S . -B build-noavx -DVECTORPULSE_ENABLE_AVX2=OFF
cmake --build build-noavx --config Release
ctest --test-dir build-noavx -C Release --output-on-failure
```

AVX2 is optional. On supported x86/x86-64 toolchains only the SIMD source is
compiled with `/arch:AVX2` (MSVC) or `-mavx2 -mfma` (GCC/Clang). Other sources
retain their baseline architecture. Runtime detection requires AVX2, FMA, and
OS support for AVX register state. A disabled kernel or unsupported CPU makes
`AVX2SearchBackend::is_supported()` return false; construction throws
`std::runtime_error`. There is no implicit fallback. Scalar remains usable.
Do not add global AVX flags when building for CPUs without AVX support.

## Select the AVX2 backend

```cpp
#include "vectorpulse/avx2_search_backend.h"
#include "vectorpulse/vector_store.h"
#include <memory>

if (vectorpulse::AVX2SearchBackend::is_supported()) {
    vectorpulse::VectorStore store{
        768, std::make_unique<vectorpulse::AVX2SearchBackend>(768)};
    // Use store.add(), store.get(), and store.search() as usual.
}
```

The SIMD loop widens float inputs to double precision, processes eight
coordinates per iteration using FMA, then handles remaining coordinates with a
scalar tail. It preserves the baseline's zero-vector behavior and avoids float
accumulation overflow for large finite inputs.

## Benchmark

The executable compares both backends using the exact same generated vectors
and queries (seed 42). No arguments selects the quick workload:

```powershell
./build/Release/vectorpulse_benchmark.exe
./build/Release/vectorpulse_benchmark.exe --vectors 100000 --dimension 768 --queries 100 --top-k 10
```

With Ninja the executable is normally `./build/vectorpulse_benchmark`.
Each backend receives one untimed warm-up query. Timings include searches and
retention of result vectors, excluding generation, insertion, checksums and
correctness comparison. All returned scores and IDs feed printed checksums.
Every Top-K ID/rank and score is compared after timing, with absolute score
tolerance `1e-6`; a mismatch returns failure. Unsupported machines still run
scalar and explicitly report AVX2 as unsupported.

Measured on Intel Core i9-14900F, Windows x64, MSVC 19.44.35228.0, CMake
3.31.6-msvc6, Release (`/O2 /Ob2 /DNDEBUG /MD /std:c++20`, default precise
floating-point semantics; `/W4 /permissive-`; kernel only `/arch:AVX2`).

| Vectors × dimensions; queries; K | Scalar latency | Scalar QPS | AVX2 latency | AVX2 QPS | Speedup |
|---|---:|---:|---:|---:|---:|
| 10,000 × 128; 100; 10 | 1.139 ms | 878.049 | 0.333 ms | 3006.624 | 3.424× |
| 100,000 × 768; 100; 10 | 69.197 ms | 14.452 | 24.312 ms | 41.131 | 2.846× |

Both measured comparisons had identical ID and score checksums and zero maximum
score error. These are single-run, same-process measurements, scalar first,
without core affinity or frequency controls; timing varies with system load,
cache state, and CPU scheduling. Both stores coexist in memory (about 614 MB of
raw vector values for the full workload, plus metadata).

Before changes, the original project built and passed 12/12 tests. Its original
scalar benchmark measured 61.242 ms/query, 16.329 QPS on the full workload and
1.149 ms/query, 869.977 QPS on the quick workload. These original timings are
recorded separately; speedup above uses the paired comparison run, not the
pre-change run. The new benchmark consumes all Top-K results, whereas the old
checksum consumed only the first result.

Raw results: [baseline](docs/benchmarks/milestone2a-baseline.txt),
[quick](docs/benchmarks/milestone2a-quick.txt),
[full](docs/benchmarks/milestone2a-full.txt).

## Validation and limitations

The AVX2-enabled Release build passes all 37 tests. Tests compare scores and
exact ID/rank order across dimensions 1–17, 31, 384, 767, 768 and 769, with
zero vectors, exact ties, all-result searches, multiple K values, validation,
retrieval, duplicate handling, and extreme finite values.

The AVX2-disabled Release build also succeeds: 13 tests pass (including
unsupported-construction behavior), 24 SIMD-dependent tests are explicitly
skipped, and none fail. Its benchmark runs scalar and prints AVX2 as unsupported.
The [disabled-build test log](docs/benchmarks/milestone2a-noavx-tests.txt) records
the complete results.

Floating-point reduction order differs from scalar, so bitwise score equality
and ranking of arbitrarily close scores cannot be guaranteed for every possible
input. Exact score ties use ascending IDs. NaN/infinity inputs have no new
contract; the inherited scalar implementation does not validate them. The
implementation remains an exhaustive scan with per-query allocation and
sorting; it does not cache norms or add concurrency. MSVC x64 is the tested
platform; GCC/Clang and non-x86 builds are supported by build guards but were
not executed here. Actual unsupported CPU hardware was not available; the
kernel-disabled configuration validates the unsupported path.

See [architecture](docs/architecture.md) for implementation details.
