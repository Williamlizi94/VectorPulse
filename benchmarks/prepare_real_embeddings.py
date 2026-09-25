"""Prepare public COCO embeddings for VectorPulse; no synthetic vectors or model inference."""
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--hdf5',type=Path,required=True)
p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--queries',type=int,default=500)
a=p.parse_args()
def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
with h5py.File(a.hdf5,'r') as f:
    if not 100000 <= len(f['train']) <= 200000: raise ValueError('Expected 100k-200k base vectors')
    if not 1 <= a.queries <= len(f['test']): raise ValueError('Invalid query count')
    indices=np.random.default_rng(42).permutation(len(f['test']))
    vectors=np.asarray(f['train'],dtype='<f4')
    all_queries=np.asarray(f['test'],dtype='<f4')
    hashes={hashlib.sha256(row.tobytes()).digest() for row in vectors}
    indices=np.sort([int(i) for i in indices if hashlib.sha256(all_queries[i].tobytes()).digest() not in hashes][:a.queries])
    if len(indices) != a.queries: raise ValueError('Too few queries after overlap exclusion')
    queries=all_queries[indices]
    assert np.isfinite(vectors).all() and np.isfinite(queries).all()
    assert (np.linalg.norm(vectors,axis=1)>0).all() and (np.linalg.norm(queries,axis=1)>0).all()
    # Verify that selected queries have no exact base overlap.
    overlap=sum(hashlib.sha256(row.tobytes()).digest() in hashes for row in queries)
    a.prefix.parent.mkdir(parents=True,exist_ok=True)
    vectors.tofile(str(a.prefix)+'.vectors.f32')
    queries.tofile(str(a.prefix)+'.queries.f32')
    metadata=dict(source='https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-i2i-512-angular.hdf5',
        source_sha256=sha(a.hdf5),source_dtype=str(f['train'].dtype),N=len(vectors),D=vectors.shape[1],queries=len(queries),
        query_indices=indices.tolist(),selection_seed=42,query_base_exact_overlap=overlap,
        unique_base_rows=len(hashes),normalization='none; lossless float16 to float32 conversion; cosine inside each backend',
        vectors_sha256=sha(str(a.prefix)+'.vectors.f32'),queries_sha256=sha(str(a.prefix)+'.queries.f32'))
    Path(str(a.prefix)+'.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print({k:v for k,v in metadata.items() if k!='query_indices'})
