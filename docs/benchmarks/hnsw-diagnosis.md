ï»¿# HNSW recall diagnosis

The diagnostic scope is 100,000 vectors, cosine similarity, K=10. No new
million-vector run is performed. The production HNSW implementation is retained;
only a friend declaration permits read-only benchmark/test inspection. Exact
CPU, AVX2 and CUDA implementations are unchanged.

## Experiment design

The full Cartesian grid uses N=100,000 and D=128 for both distributions,
M={16,32,48}, efConstruction={100,200,400}, efSearch={40,80,160,320}:
18 independent graphs and 72 measured query configurations. Two additional
N=100,000, D=768 controls use M=16/efConstruction=200 to check the dimension
of the earlier million-vector run directly. Each graph uses seed 42, 100
held-out queries, one warm-up pass and three measured repetitions. These
are 100 unique queries, not 300 independent accuracy observations.

- **Random:** independent FP32 components uniform in [-1,1], matching the
  earlier benchmark distribution. Stored vectors need not be pre-normalized;
  the existing cosine helper computes both norms.
- **Clustered:** 100 Gaussian unit-vector centers. Each vector/query is a
  center plus independent Gaussian noise with component standard deviation
  0.25/sqrt(D), then L2 normalized. Cluster IDs cycle through 0..99, interleaving
  clusters during insertion; there is one independent query per cluster.
  This is a simple embedding-like synthetic model, not a real embedding corpus.

Every graph is evaluated against the existing scalar VectorIndex, with the same
FP32 data and held-out queries. Dataset/query hashes in the logs must agree
across all nine configurations of each distribution. Each returned result is
checked for count, unique valid IDs, descending cosine/ID order, and exact
scalar score bits. Recall@10 is mean ID intersection with scalar Top-10 / 10.
No reranking, rescoring of returned results, or exact-search fallback is used.

Latency covers the C++ search call and returned-result construction. Validation,
recall, graph audits, full-breadth controls and result destruction are outside
timing. efSearch configuration order is shuffled between measured rounds with
seed 2026. The reference ground-truth timings are one pass, not an equally
warmed exact performance benchmark.

To keep experiments bounded, three sweep processes use Windows logical CPU
0/2/4; the two 768-dimensional controls use 6/8. An optional independent
reference uses logical CPU 10. The processes can build concurrently, sharing
cache/memory bandwidth. These are **diagnostic latencies under concurrent
work**, not isolated hardware performance rankings or direct comparisons with
the earlier million-vector timings. Build times also include this contention.
Hardware: i9-14900F, Windows 11 x64, MSVC Release; no CUDA queries are needed.

## Implementation inspection

| Area | Inspection and verification |
| --- | --- |
| Construction | Existing nodes are searched before staged reciprocal updates are committed. Each new node selects up to M neighbors, then adds the predecessor if needed. Upper degree cap M and base cap 2*M are audited. Node IDs, dimensions, entry point and maximum layer must agree. |
| Layer traversal | Promotion probability is 1/M, matching the geometric hierarchy. Greedy traversal only follows same-layer links and strictly improves the score/ID ordering. Search descends to layer 0; graph audits check layer membership and entry reachability. |
| Candidate queues | The frontier is best-first and retained results are worst-first. The stopping comparison is in the correct similarity direction. Independent sorted-list traversal tests reproduce heap-search results across small/large ef, negative similarities and zero-query ties. |
| Neighbor selection | Candidates are processed by descending cosine. A candidate is rejected as redundant if it is at least as similar to an already selected neighbor as to the center. Rejected candidates backfill the list. Direct tests check diversity and backfill. |
| Cosine normalization | All graph/query distances call the unchanged cosine helper: double dot/norm accumulation and FP32 score. No normalized-dot-product shortcut is used. Positive power-of-two scaling preserves search results in regression tests; zero and finite-extreme inputs are covered by existing tests. |
| Connectivity | Every edge is checked for range, uniqueness, no self-loop and valid layer. The insertion-order predecessor/successor chain must be present. Directed reachability is measured per layer, plus base reachability with all index-adjacent edges excluded. Three held-out full-breadth queries per graph must match exact Top-10. |

The structure follows the [HNSW paper](https://arxiv.org/abs/1603.09320), including
its optional backfill of pruned candidates. The mandatory base chain is a local
extension: it uses up to two of 2*M adjacency slots and may influence quality.
Removing index-adjacent edges for reachability is only an inspection, not a
construction ablation; it also removes naturally selected adjacent edges.
Therefore these measurements cannot isolate the chain's causal effect on recall.
Pruned links need not remain reciprocal; asymmetry alone is not a correctness bug.

The pinned [hnswlib 0.8.0 implementation](https://github.com/nmslib/hnswlib/tree/v0.8.0)
is an additional recall control on the identical exported FP32 vectors, queries
and scalar ground truth. It builds single-threaded with M=16, efConstruction=200,
seed 42 and sweeps the same efSearch values. It uses its own graph construction,
level RNG and FP32 normalization; graph identity or bit-identical scores are not
expected. This control compares recall, not C++ versus Python-call latency.
It is optional tooling and adds no dependency to the VectorPulse library.

## Reproduction

```powershell
cmake --build build --config Release --target vectorpulse_hnsw_diagnostics
python benchmarks/run_hnsw_diagnostics.py --binary build/Release/vectorpulse_hnsw_diagnostics.exe --output out/hnsw-diagnostics-100k --workers 3
```

Use `--workers 1` for serial timings. The runner refuses more than 100,000
vectors. Individual configurations can be run directly:

```powershell
./build/Release/vectorpulse_hnsw_diagnostics.exe --dataset random --vectors 100000 --dimension 768 --m 16 --ef-construction 200 --queries 100 --export-prefix out/random-768
./build/Release/vectorpulse_hnsw_diagnostics.exe --dataset clustered --vectors 100000 --dimension 768 --m 16 --ef-construction 200 --queries 100 --export-prefix out/clustered-768
```

The export prefix writes native FP32 arrays (little-endian on this measured
machine) and scalar Top-10 IDs for the optional reference check:

```powershell
python -m pip install numpy hnswlib==0.8.0
python benchmarks/compare_hnsw_reference.py --prefix out/random-768 --dimension 768
```

Exports live under ignored `out/`. The 18-graph parameter sweep is incomplete:
9 graphs completed, 3 stopped during construction, and 6 have no result.
Both 768-dimensional controls completed. Their raw logs and the subsequent
real-embedding comparison are retained with the
[real-embedding report](hnsw-real-embeddings.md). No million-vector experiment is part of this workflow.
