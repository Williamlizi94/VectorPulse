ï»¿"""Optional independent recall control; requires hnswlib==0.8.0 and NumPy.

Consumes FP32 vectors and scalar truth exported by vectorpulse_hnsw_diagnostics.
This adds no dependency to VectorPulse. No output is used by production search.
"""
import argparse
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import hnswlib
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prefix", type=Path, required=True)
parser.add_argument("--dimension", type=int, required=True)
parser.add_argument("--m", type=int, default=16)
parser.add_argument("--ef-construction", type=int, default=200)
args = parser.parse_args()
if version("hnswlib") != "0.8.0":
    raise RuntimeError("This control is pinned to hnswlib 0.8.0")
prefix = str(args.prefix)
vectors = np.fromfile(prefix + ".vectors.f32", dtype="<f4").reshape(-1, args.dimension)
queries = np.fromfile(prefix + ".queries.f32", dtype="<f4").reshape(-1, args.dimension)
truth = np.loadtxt(prefix + ".truth.csv", delimiter=",", dtype=np.uint64, ndmin=2)
assert truth.shape == (len(queries), 10)
index = hnswlib.Index(space="cosine", dim=args.dimension)
index.init_index(max_elements=len(vectors), ef_construction=args.ef_construction, M=args.m, random_seed=42)
index.set_num_threads(1)
start = perf_counter()
index.add_items(vectors, np.arange(len(vectors)), num_threads=1)
build_s = perf_counter() - start
print(f"hnswlib={version('hnswlib')},N={len(vectors)},D={args.dimension},M={args.m},"
      f"efConstruction={args.ef_construction},seed=42,threads=1,build_s={build_s:.6f}", flush=True)
print("efSearch,Recall@10", flush=True)
for ef in (40, 80, 160, 320):
    index.set_ef(ef)
    labels, distances = index.knn_query(queries, k=10, num_threads=1)
    if not np.isfinite(distances).all():
        raise RuntimeError("nonfinite reference scores")
    recall = np.mean([len(set(a) & set(b))/10 for a, b in zip(labels, truth)])
    print(f"{ef},{recall:.6f}", flush=True)
