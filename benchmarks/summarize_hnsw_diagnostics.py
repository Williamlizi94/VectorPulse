ï»¿"""Collect a completed diagnostic grid and its two 768D controls."""
import argparse
import csv
import io
from pathlib import Path
import shutil

def read_log(path):
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    config = dict(part.split("=", 1) for part in lines[0].split(",")[1:])
    start = next(i for i, line in enumerate(lines) if line.startswith("dataset,N,D,M,"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    assert len(rows) == 4 and {r["efSearch"] for r in rows} == {"40", "80", "160", "320"}, path
    assert "FULL_BREADTH,queries=3,recall=1" in lines, path
    layers = []
    for line in lines:
        if line.startswith("LAYER,"):
            pieces = line.split(",")
            layer = dict(part.split("=", 1) for part in pieces[2:])
            layer["level"] = pieces[1]
            layers.append(layer)
    assert layers and layers[0]["reachable"] == config["N"], path
    graph = next(line for line in lines if line.startswith("GRAPH,"))
    graph = dict(part.split("=", 1) for part in graph.split(",")[1:])
    truth = next(line for line in lines if line.startswith("TRUTH,"))
    truth = dict(part.split("=", 1) for part in truth.split(",")[1:])
    for row in rows:
        for key in ("N", "D", "M", "efConstruction", "dataset"):
            assert row[key] == config[key], (path, key)
    return {"path": path, "config": config, "rows": rows, "layers": layers, "graph": graph, "truth": truth}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for dataset in ("random", "clustered"):
        for m in (16, 32, 48):
            for efc in (100, 200, 400):
                records.append(read_log(args.sweep / f"{dataset}-n100000-d128-m{m}-efc{efc}.txt"))
        records.append(read_log(args.controls / f"{dataset}-n100000-d768-m16-efc200.txt"))
    for dataset in ("random", "clustered"):
        cells = [r for r in records if r["config"]["dataset"] == dataset and r["config"]["D"] == "128"]
        assert len(cells) == 9
        assert len({(r["config"]["data_hash"], r["config"]["query_hash"]) for r in cells}) == 1
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    all_rows = [row for record in records for row in record["rows"]]
    assert len(all_rows) == 80
    with (args.output / "results.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    for record in records:
        shutil.copyfile(record["path"], logs / record["path"].name)
    report = ["# HNSW diagnostic measurements", "",
        "Each cell below is **Recall@10 (%) / mean query latency (ms)**. N=100,000, K=10.",
        "See [methodology and findings](../hnsw-diagnosis.md) for the concurrent CPU",
        "execution conditions and interpretation. All 80 query configurations",
        "passed result validation; all 20 graphs passed structural audits and",
        "three full-breadth scalar-equality queries each.", "",
        "[Complete CSV](results.csv) also contains build time, p95 and QPS.", ""]
    for dataset in ("random", "clustered"):
        report += [f"## {dataset.capitalize()}, 128 dimensions", "",
            "| M | efConstruction | efSearch=40 | 80 | 160 | 320 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for m in (16, 32, 48):
            for efc in (100, 200, 400):
                record = next(r for r in records if r["config"]["dataset"] == dataset
                    and r["config"]["D"] == "128" and r["config"]["M"] == str(m)
                    and r["config"]["efConstruction"] == str(efc))
                cells = [f"{float(row['Recall@10'])*100:.1f}% / {float(row['mean_ms']):.3f}"
                         for row in record["rows"]]
                report.append(f"| {m} | {efc} | " + " | ".join(cells) + " |")
        report.append("")
    report += ["## 768-dimensional controls, M=16 / efConstruction=200", "",
        "| Dataset | efSearch=40 | 80 | 160 | 320 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for record in records:
        if record["config"]["D"] == "768":
            cells = [f"{float(row['Recall@10'])*100:.1f}% / {float(row['mean_ms']):.3f}"
                     for row in record["rows"]]
            report.append("| " + record["config"]["dataset"] + " | " + " | ".join(cells) + " |")
    report += ["", "## Independent hnswlib 0.8.0 recall control", "",
        "M=16, efConstruction=200. Each cell is VectorPulse / hnswlib Recall@10.",
        "The graph RNG, FP32 arithmetic, normalization and neighbor heuristics differ;",
        "this is a diagnostic comparison, not a bit-identity requirement.", "",
        "| Dataset | Dimensions | efSearch=40 | 80 | 160 | 320 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for record in records:
        if record["config"]["M"] != "16" or record["config"]["efConstruction"] != "200": continue
        path = record["path"].with_suffix(".reference.txt")
        lines = path.read_text().splitlines()
        reference = list(csv.DictReader(lines[1:]))
        assert len(reference) == 4
        shutil.copyfile(path, logs / path.name)
        cells = []
        for own, other in zip(record["rows"], reference):
            assert own["efSearch"] == other["efSearch"]
            cells.append(f"{float(own['Recall@10'])*100:.1f}% / {float(other['Recall@10'])*100:.1f}%")
        report.append(f"| {record['config']['dataset']} | {record['config']['D']} | " + " | ".join(cells) + " |")
    report += ["", "## Graph audit", "",
        "Base directed reachability is 100,000/100,000 for every graph. The last",
        "column excludes all edges between adjacent insertion indices, including",
        "any naturally selected adjacent edges; this is not a chain ablation.", "",
        "| Dataset | D | M | efConstruction | Mean base degree | Reciprocal edges | Base reachable excluding adjacent indices |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        c, g = r["config"], r["graph"]
        report.append(f"| {c['dataset']} | {c['D']} | {c['M']} | {c['efConstruction']} | "
            f"{float(r['layers'][0]['mean_degree']):.2f} | {float(g['reciprocal_fraction'])*100:.1f}% | "
            f"{g['base_reachable_without_adjacent_ids']} |")
    report += ["", "Raw per-configuration logs under [logs/](logs/) retain dataset/query hashes,",
               "per-layer node/degree/reachability statistics and exact-score geometry.", ""]
    (args.output / "results.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Validated {len(records)} graphs, {len(all_rows)} measured cells, identical per-dataset hashes.")
if __name__ == "__main__":
    main()
