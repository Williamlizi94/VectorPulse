# HNSW real-embedding diagnostic - 2026-09-25

**Result: VectorPulse HNSW behaves similarly to hnswlib on this real embedding
workload.** At efSearch=80 it reaches 99.62% Recall@10 in 0.757 ms/query;
hnswlib reaches 99.54%. No correctness bug was found in this validation.
Search implementations were unchanged during validation. No new 1M benchmark
was run. These measurements were completed before the v0.3.0 release preparation.

## Data and protocol

[COCO-I2I release](https://github.com/fabiocarrara/str-encoders/releases/tag/v0.1.3):
113,287 CLIP ViT-B/16 image embeddings, 512 dimensions, cosine similarity, K=10.
The published random split has 10,000 held-out image queries. We select 500 with
a seed-42 permutation, excluding exact base/query duplicates. Released float16
values are converted losslessly to FP32 without external normalization. The
113,275 distinct base vectors retain all 113,287 IDs, including duplicates.
[Metadata](real-embeddings/metadata.json) preserves selected row indices and
SHA-256 hashes. This is real image embedding data, not synthetic clustering;
it does not establish performance on text embeddings or all CLIP workloads.

All backends receive identical vectors, insertion order and queries. Scalar
exhaustive cosine recomputes ground truth. M=16, efConstruction=200, seed=42;
hnswlib 0.8.0 uses the same parameters, cosine and one build/query thread.
Its graph RNG and FP32 normalization differ from VectorPulse's scalar cosine.

One full warm-up and three measured passes give 1,500 latency samples from
**500 unique queries**. Backends run sequentially with one index alive at a
time; efSearch order is shuffled each measured round. Build includes copies,
insertion and CUDA's eager upload. Query timing includes returned-result
construction, excludes validation/destruction; QPS=1000/mean_ms.
Hardware: Windows x64, i9-14900F, RTX 4080 SUPER (16 GiB), MSVC Release,
CUDA runtime 12.9, driver 610.60. CPU/GPU clocks and other system activity were
not controlled; exact backend order was fixed.

## Measurements

| Backend | Build s | Mean ms | QPS | Recall@10 | RSS after build GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact scalar (ground truth) | 0.077 | 43.376 | 23.05 | 100.00% | 0.476 |
| Exact AVX2 | 0.086 | 15.606 | 64.08 | 100.00% | 2.588 |
| Exact AVX2, 16 threads | 0.076 | 5.759 | 173.63 | 100.00% | 2.588 |
| Exact CUDA, persistent block FP64 | 0.185 | 3.487 | 286.79 | 100.00% | 2.699 |
| HNSW ef=40 | 248.677 | 0.460 | 2171.97 | 98.94% | 2.739 |
| HNSW ef=80 | 248.677 | 0.757 | 1321.46 | 99.62% | 2.739 |
| HNSW ef=160 | 248.677 | 1.245 | 803.09 | 99.92% | 2.739 |
| HNSW ef=320 | 248.677 | 2.192 | 456.26 | 99.96% | 2.739 |

The four HNSW rows share **one 248.677-second build**. RSS is whole-process,
including the input and retained allocator/runtime memory from earlier stages;
it is not isolated index storage. Process peak: 2.916 GiB. CUDA database:
221.264 MiB; the raw FP32 base payload is also 221.264 MiB.
[Full CSV](real-embeddings/results.csv) includes p95 and before/after memory.

## Independent recall control

| efSearch | VectorPulse | hnswlib 0.8.0 | Difference (percentage points) |
| ---: | ---: | ---: | ---: |
| 40 | 98.94% | 98.16% | +0.78 |
| 80 | 99.62% | 99.54% | +0.08 |
| 160 | 99.92% | 99.86% | +0.06 |
| 320 | 99.96% | 99.92% | +0.04 |

Both show the same high-recall curve, with a maximum observed gap of 0.78
percentage points. This supports similar recall behavior on the measured
queries, not statistical equivalence across datasets or build seeds.
VectorPulse's build took 248.677 s versus hnswlib's 35.115 s (about **7.1x longer**).
The reference times add_items only, while VectorPulse also includes index
construction and progress output; the comparison is indicative. hnswlib query
latency was not benchmarked. [Reference output](real-embeddings/hnswlib.txt).

## Earlier findings in context

Entries are VectorPulse / hnswlib Recall@10, M=16, efConstruction=200.
The old synthetic runs used 100 queries and concurrent build processes;
their timings should not be compared directly with this serial experiment.

| Dataset | ef=40 | ef=80 | ef=160 | ef=320 |
| --- | ---: | ---: | ---: | ---: |
| Random, 100k x 768 | 3.4% / 3.7% | 6.4% / 6.2% | 12.5% / 11.6% | 21.8% / 21.1% |
| Clustered, 100k x 768 | 92.2% / 91.8% | 97.7% / 97.9% | 99.9% / 99.8% | 100% / 99.8% |
| Real COCO, 113,287 x 512 | 98.94% / 98.16% | 99.62% / 99.54% | 99.92% / 99.86% | 99.96% / 99.92% |

The earlier [1M x 768 random run](hnsw-large.md) had only 0.6?3.5% recall;
it was not rerun, and has no matching 1M hnswlib control. Together, the completed
100k controls and COCO result support data geometry as a major cause of the
random-data failure, rather than a VectorPulse-specific recall defect. The old
18-graph parameter sweep remains incomplete; no missing cells are inferred.
Retained random/clustered raw evidence is under [real-embeddings/](real-embeddings/).

## Recommended starting configuration

Use **M=16, efConstruction=200, efSearch=80** as a balanced starting point for
structured embeddings. On COCO this gives 99.62% recall at 0.757 ms, about 4.6x
the measured exact CUDA QPS. Use **efSearch=160** when recall is more important:
99.92% on COCO and 99.9% on the earlier clustered control. ef=320 buys only
0.04 percentage points over 160 on COCO while increasing latency by 76%.
M and efConstruction were held fixed here: this is a validated starting point,
not a demonstrated optimum. Tune on held-out application queries and a chosen
recall target; these settings do not solve uniform random high-dimensional data.
The library's existing default efSearch=50 has not been changed.

## Validation and reproduction

Every measured/warm-up result passed count, unique valid IDs, finite scores and
score/ID sorting checks. HNSW returned scores equal scalar cosine exactly;
AVX2/CUDA scores passed 1e-6 absolute tolerance and all exact backends had 100%
strict-ID Recall@10. Alternative IDs at tied scores are not specially credited.
[Source hashes](real-embeddings/source-preservation.json) confirm all 24 recorded
implementation/header files are unchanged. **CUDA Release regression: 406/406 tests passed.** The new input harness also
passed a small real-data full-breadth exact-equality check and rejected truncated,
oversized and nonfinite inputs. See [regression log](real-embeddings/cuda-tests.txt)
and [harness checks](real-embeddings/harness-checks.txt).

Install numpy, h5py and hnswlib==0.8.0 in a diagnostic environment, download the
linked release HDF5, and build vectorpulse_hnsw_benchmark with CUDA if available:

```powershell
python benchmarks/prepare_real_embeddings.py --hdf5 out/real-embeddings/coco-i2i-512-angular.hdf5 --prefix out/real-embeddings/coco
python benchmarks/run_real_embeddings.py --binary out/build-cuda/vectorpulse_hnsw_benchmark.exe --prefix out/real-embeddings/coco --output out/real-embeddings/repeat
```

The runner verifies vector hashes, caps the dataset at 200k and runs VectorPulse
then hnswlib sequentially. Dataset preparation requires Python 3.11+.
