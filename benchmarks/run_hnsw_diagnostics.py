ï»¿"""Run the 100K HNSW diagnostic grid (no million-vector configuration).

Each of three workers owns a CPU affinity slot on Windows. Measurements include
normal concurrent-build/cache contention; this is a diagnosis, not an isolated
performance ranking. Each process generates identical seed-42 data and truth.
"""
import argparse
import concurrent.futures
import ctypes
import itertools
from pathlib import Path
import subprocess
import sys
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vectors", type=int, default=100000)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if args.vectors < 10 or args.vectors > 100000:
        parser.error("diagnostic runner is limited to 10..100000 vectors")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = list(itertools.product(("random", "clustered"), (16, 32, 48), (100, 200, 400)))
    binary = args.binary.resolve()
    started = time.perf_counter()
    def worker(slot):
        for dataset, m, efc in tasks[slot::args.workers]:
            stem = args.output / f"{dataset}-n{args.vectors}-d{args.dimension}-m{m}-efc{efc}"
            command = [str(binary), "--dataset", dataset, "--vectors", str(args.vectors),
                       "--dimension", str(args.dimension), "--queries", str(args.queries),
                       "--m", str(m), "--ef-construction", str(efc)]
            if m == 16 and efc == 200:
                command += ["--export-prefix", str(stem)]
            with stem.with_suffix(".txt").open("w") as output, stem.with_suffix(".progress.txt").open("w") as progress:
                process = subprocess.Popen(command, stdout=output, stderr=progress)
                if sys.platform == "win32":
                    set_affinity = ctypes.windll.kernel32.SetProcessAffinityMask
                    set_affinity.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                    set_affinity.restype = ctypes.c_int
                    if not set_affinity(int(process._handle), 1 << (2 * slot)):
                        process.terminate()
                        process.wait()
                        raise OSError("cannot set benchmark CPU affinity")
                result = process.wait()
            if result:
                raise RuntimeError(f"{stem.name} failed ({result}); see logs")
            print(f"DONE {stem.name} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, range(args.workers)))

if __name__ == "__main__":
    main()
