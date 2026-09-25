# CPU HNSW index

Include `<vectorpulse/hnsw_index.h>` and link `VectorPulse::vectorpulse`.
`HnswIndex` is an independent, append-only C++20 approximate cosine index.
The existing exact `VectorIndex`, storage, persistence, and CPU/AVX2/CUDA
search algorithms are unchanged.

## API

```cpp
#include <vectorpulse/hnsw_index.h>
#include <vector>

int main() {
    vectorpulse::HnswConfig options{
        .M = 16, .efConstruction = 200, .efSearch = 50, .seed = 42};
    vectorpulse::HnswIndex index{3, options};
    index.add("east", {1.0F, 0.0F, 0.0F});
    index.add("north", {0.0F, 1.0F, 0.0F});
    const std::vector<float> query{1.0F, 0.0F, 0.0F};
    auto results = index.search(query, 2);
    auto wider = index.search(query, 2, 100); // Per-query efSearch override.
    index.set_ef_search(100);                // Change the default breadth.
    return index.size() == 2 && index.dimension() == 3 ? 0 : 1;
}
```

| Parameter | Default | Contract |
| --- | ---: | --- |
| dimension | required | Positive, fixed for the lifetime of the index |
| M | 16 | At least 2; must permit representing 2*M in size_t |
| efConstruction | 200 | At least M |
| efSearch | 50 | Positive; effective breadth is min(size, max(K, efSearch)) |
| seed | 42 | Seed for geometric level sampling |

Larger M allows more graph links and increases memory/build cost.
Larger efConstruction explores more insertion candidates and generally improves
graph quality at higher build cost. Larger efSearch explores more query
candidates and generally improves recall at higher latency. Recall need not
increase monotonically for every individual query.

`add(std::string, std::vector<float>)` owns a copy/move of each ID and vector.
IDs must be unique; empty strings and embedded NUL bytes are supported.
Dimension mismatches, duplicate IDs, NaN/infinity components and invalid
configuration throw `std::invalid_argument`. Failed insertions leave the
graph, ID map and level generator unchanged.

`search(std::span<const float>, size_t)` returns `std::vector<SearchResult>`,
containing string `id` and float `score`. Returned candidates are ordered by
descending cosine score, then ascending ID for exact ties. Candidate selection
is approximate; an omitted vector may score higher than a returned vector.
Scoring calls the existing cosine helper (double accumulation, FP32 result);
zero vectors score zero and negative similarities are retained. Results contain
min(K, size()) distinct IDs. Empty indexes and K=0 return empty results;
query validation still applies. Oversized K returns all entries.

`config()` returns a copy of the configuration. M, efConstruction and seed are
fixed at construction; only the default efSearch can change. Per-query overrides
do not mutate the index. Read-only searches may run concurrently; callers must
exclude add/set_ef_search from concurrent operations.

## Implementation

The implementation follows the layered graph, greedy upper-layer descent,
bounded best-first layer search and diversity-based neighbor selection described
by [Malkov and Yashunin](https://arxiv.org/abs/1603.09320).
New nodes sample geometric levels with promotion probability 1/M (capped at 32).
Layer search uses candidate/result heaps and a per-query visited set; it does
not clear an O(N) visited array or scan the entire vector store before traversal.
Pruned candidates backfill neighbor lists when diversity alone leaves them short.
Upper-layer adjacency is bounded by M and base adjacency by 2*M.

One explicit extension reserves predecessor/successor edges from insertion order
within the base-layer degree budget. This keeps the base graph reachable even
for all-zero or duplicate vectors after repeated pruning. These edges can occupy
two of the 2*M slots and affect the recall/performance trade-off, especially at
small M. Other graph links need not remain reciprocal after pruning.

Full breadth (efSearch >= size()) traverses the connected base graph and matches
exact scalar Top-K, including score bits and ID tie order. This uses graph
traversal, with no exact-search fallback. A fixed seed, input values and insertion
order produce repeatable builds/results on the same floating-point environment;
cross-platform floating-point bit identity is not promised.

HNSW currently has no deletion, persistence, CUDA backend or Python binding.
Existing exact persistence and Python APIs continue to work.

## Tests and benchmark

The tests cover constructor validation, empty/singleton indexes, K boundaries,
dimensions 1/3/128/769, strings, ties, zero vectors, finite extremes, invalid
insertions, deterministic builds, post-query insertion, concurrent reads,
degenerate graph reachability and exact-ground-truth recall. Full breadth checks
require exact IDs, order and score bits. A seeded held-out dataset also exercises
approximate search with a recall threshold.

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release -R Hnsw --output-on-failure
./build/Release/vectorpulse_hnsw_benchmark.exe --vectors 10000 --dimension 64 --queries 100 --top-k 10 --m 16 --ef-construction 200 --ef-search 10,20,40,80,160,320 --warmups 1 --repetitions 3 --seed 42
```

The benchmark compares exact scalar, AVX2, multithreaded AVX2, available
persistent FP64 CUDA, and HNSW on identical data and held-out queries.
Only one backend is alive at a time to bound memory; scalar ground truth is
retained. One HNSW graph is reused across the efSearch sweep. Query timing
excludes validation and recall calculation, and CUDA build time includes eager
database upload.

Use `--preset large` for one million vectors, 768 dimensions, K=10 and
efSearch=40,80,160,320. `--threads 16` configures the threaded exact backend.
Output includes build time, mean/median/p95 latency, QPS, measured Recall@K,
whole-process resident memory and persistent GPU database bytes. Unavailable
AVX2/CUDA backends are explicitly skipped.

See [large-scale methodology and results](benchmarks/hnsw-large.md) for the
complete timing, memory and recall definitions. The [initial scalar/HNSW
measurements](benchmarks/hnsw-results.md) remain a historical baseline.

The [100K recall diagnosis](benchmarks/hnsw-diagnosis.md) sweeps graph parameters
on random and clustered data and audits construction/search invariants.
