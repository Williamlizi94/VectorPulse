"""Run the prepared real-embedding comparison sequentially (no million-vector run)."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--m',type=int,default=16)
p.add_argument('--ef-construction',type=int,default=200)
a=p.parse_args()
meta=json.loads(Path(str(a.prefix)+'.metadata.json').read_text())
if not 100000<=meta['N']<=200000: p.error('Real-data diagnostic is limited to 100k-200k vectors')
for suffix,key in [('.vectors.f32','vectors_sha256'),('.queries.f32','queries_sha256')]:
    with open(str(a.prefix)+suffix,'rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=meta[key]:
            raise ValueError('Prepared data hash mismatch: '+suffix)
a.output.mkdir(parents=True,exist_ok=True)
command=[str(a.binary.resolve()),'--input-prefix',str(a.prefix),'--vectors',str(meta['N']),
    '--dimension',str(meta['D']),'--queries',str(meta['queries']),'--top-k','10','--m',str(a.m),
    '--ef-construction',str(a.ef_construction),'--ef-search','40,80,160,320','--threads','16',
    '--warmups','1','--repetitions','3']
with (a.output/'vectorpulse.txt').open('w') as out,(a.output/'vectorpulse-progress.txt').open('w') as err:
    subprocess.run(command,stdout=out,stderr=err,check=True)
with (a.output/'hnswlib.txt').open('w') as out:
    subprocess.run([sys.executable,str(Path(__file__).with_name('compare_hnsw_reference.py')),
        '--prefix',str(a.prefix),'--dimension',str(meta['D']),'--m',str(a.m),
        '--ef-construction',str(a.ef_construction)],stdout=out,check=True)
