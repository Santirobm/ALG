import csv
from collections import defaultdict

data = defaultdict(list)
with open("alg_hnsw/results/at_transit_ablation.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = (r["Scenario"], int(r["Selectivity"]), r["Variant"])
        data[key].append({
            "R10": float(r["Recall10"]),
            "R100": float(r["Recall100"]),
            "p50": float(r["p50_us"]),
            "p95": float(r["p95_us"]),
            "qps": float(r["QPS"]),
            "exp": float(r["ExpandedNodes"]),
            "transit_pct": float(r["PctTransit"]),
            "leaks": int(r["Leaks"])
        })

print(f"{'Scenario':<14} | {'Sel':<4} | {'Variant':<10} | {'Recall@10':<9} | {'Recall@100':<10} | {'p50 (us)':<8} | {'p95 (us)':<8} | {'QPS':<7} | {'Expanded':<8} | {'Transit%':<8} | {'Leaks'}")
print("-" * 115)

for (sc, sel, var), rows in sorted(data.items()):
    n = len(rows)
    r10 = sum(x["R10"] for x in rows) / n
    r100 = sum(x["R100"] for x in rows) / n
    p50 = sum(x["p50"] for x in rows) / n
    p95 = sum(x["p95"] for x in rows) / n
    qps = sum(x["qps"] for x in rows) / n
    exp = sum(x["exp"] for x in rows) / n
    trans = sum(x["transit_pct"] for x in rows) / n
    leaks = sum(x["leaks"] for x in rows)
    print(f"{sc:<14} | {sel:2d}%  | {var:<10} | {r10*100:6.2f}%   | {r100*100:6.2f}%    | {p50:7.1f}  | {p95:7.1f}  | {qps:7.1f} | {exp:8.1f} | {trans:6.1f}%  | {leaks}")
