# Architecture

## Components

```text
VectorStore owns SearchBackend
                 ├── ScalarSearchBackend (default, unchanged Milestone 1 baseline)
                 └── AVX2SearchBackend   (optional, single-threaded Milestone 2A)
```

`SearchBackend` defines dimension, size, insertion, retrieval and Top-K search.
`VectorStore` remains the unchanged facade and accepts a backend through its
existing `unique_ptr<SearchBackend>` constructor argument. Each backend owns
its entries and ID index. No facade validation or forwarding logic is copied
into the new backend. The AVX2 backend mirrors the scalar storage and sorting
contract; preserving the baseline unchanged avoids a storage refactor in this
milestone. It uses no external math or vector-search libraries.

Both backends scan all vectors in O(n*d), build `SearchResult` values, and use
`partial_sort` for Top-K or `sort` for all entries. Ranking is descending cosine
score, then ascending ID for exact ties. Dimensions must match; zero dimension
is rejected; duplicate IDs are rejected without replacement. Missing IDs throw.
Query validation precedes handling zero K and empty stores. Zero-magnitude
vectors have score zero, and K greater than size returns every entry.

## SIMD boundary and dispatch

`avx2_search_backend.cpp` implements the public backend and checks availability
in its constructor. `avx2_support.cpp` is compiled for the baseline architecture:

- MSVC checks CPUID leaf 1 for AVX, FMA and OSXSAVE, then XGETBV for XMM/YMM state,
  and leaf 7 for AVX2. XGETBV is executed only after OSXSAVE is confirmed.
- GCC/Clang use CPU feature builtins for AVX2 and FMA, including OS AVX support.
- If no kernel was compiled, availability is false and a stub cannot execute SIMD.

`avx2_kernel.cpp` alone receives AVX2/FMA compile flags, conditionally on x86
and compiler flag support. `VECTORPULSE_ENABLE_AVX2=OFF` removes it entirely.
The internal kernel accepts validated equal-sized spans and is only called by
a successfully constructed backend. Unsupported construction throws a clear
runtime error; no silent fallback occurs. No static initializer executes SIMD.
The default scalar backend never calls the kernel.

The kernel processes eight float coordinates with two groups of four doubles:
unaligned loads, float-to-double conversion, independent FMA accumulators for
dot product and both squared norms, horizontal sums, then a scalar tail of
zero to seven elements. Unaligned loads require no changes to vector storage.
Double products/accumulation preserve the baseline's range and precision.
The final cosine formula and zero-norm return match scalar. Norms are recomputed
per comparison, as in the baseline, so this milestone isolates SIMD acceleration.

Reduction order/FMA can change low-order bits. Tests use absolute score tolerance
1e-6, appropriate for cosine's bounded finite result, and independently require
exact IDs at every rank. Tested ties and zero-vector ranks match; arbitrary
near-ties can differ due to rounding. Nonfinite inputs retain the baseline's
unspecified behavior and are outside the correctness claim.

## Benchmark and validation

The benchmark generates each vector once and copies it to both stores, then
shares the same query list. Seed, dimensions, vector/query counts and K are
printed. It warms each backend with one query, times complete searches using
`steady_clock`, retains results, then computes score/ID checksums and validates
every rank against scalar outside timing. Unsupported platforms print the reason
and still report scalar. Default workload is 10,000 vectors, 128 dimensions,
100 queries, K=10; the full workload is 100,000 vectors, 768 dimensions,
100 queries, K=10.

On Intel Core i9-14900F with MSVC 19.44.35228.0 Release x64, the full paired run
measured scalar 69.197 ms/query (14.452 QPS), AVX2 24.312 ms/query (41.131 QPS),
2.846x speedup. The quick paired run measured 1.139 versus 0.333 ms/query,
3.424x speedup. Every returned ID/rank and score matched in both runs.
[README](../README.md) records flags, methodology and limitations;
[raw results](benchmarks/milestone2a-full.txt) preserve full precision checksums.

All 37 enabled-build tests pass, including the original 12. Parameterized tests
cover every tail length, sub-eight dimensions, 384/768-dimensional embeddings,
zero queries/vectors, random finite vectors, full rankings and multiple K values.
Additional tests cover exact ties, storage contracts, unsupported construction,
and extreme finite float magnitudes.

The kernel-disabled Release configuration builds successfully: 13 tests pass,
24 SIMD-dependent tests skip, and zero fail. Its benchmark still executes scalar
and clearly reports the unavailable backend. This exercises the compiled-out
path, not an emulation of a CPU missing individual AVX/FMA/OSXSAVE features.

No CUDA, multithreading or other later-milestone functionality is included.
