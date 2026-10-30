import csv

def load_hnswlib():
    data = {}
    with open("alg_hnsw/hito6/results/hnswlib_summary.csv", "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["Scenario"], int(row["Selectivity"]), int(row["efSearch"]))
            data[key] = {
                "R10": float(row["Recall10"]),
                "R100": float(row["Recall100"]),
                "p50": float(row["p50_us"]),
                "p95": float(row["p95_us"]),
                "qps": float(row["QPS"]),
                "mem": float(row["IndexMemMiB"]),
                "leaks": int(row["Leaks"])
            }
    return data

def load_alg():
    data = {}
    with open("alg_hnsw/hito6/results/alg_summary.csv", "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["Scenario"], int(row["Selectivity"]), row["Variant"])
            data[key] = {
                "R10": float(row["Recall10"]),
                "R100": float(row["Recall100"]),
                "p50": float(row["p50_us"]),
                "p95": float(row["p95_us"]),
                "qps": float(row["QPS"]),
                "mem": float(row["IndexMemMiB"]),
                "leaks": int(row["Leaks"])
            }
    return data

def load_exact():
    data = {}
    with open("alg_hnsw/hito6/results/exact_scan_summary.csv", "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["Scenario"], int(row["Selectivity"]))
            data[key] = {
                "p50": float(row["p50_us"]),
                "p95": float(row["p95_us"]),
                "qps": float(row["QPS"]),
                "count": int(row["EligibleCount"])
            }
    return data

hnsw = load_hnswlib()
alg = load_alg()
exact = load_exact()

print("=" * 115)
print("COMPARATIVA TRIPARTITA: ALG H2-AT T1-A vs hnswlib vs ExactScan (Semilla 1)")
print("=" * 115)

for sc in ["R_Local", "D_Adversarial"]:
    print(f"\n============================ ESCENARIO {sc} ============================")
    for sel in [1, 5, 10, 25, 50]:
        print(f"\n--- Selectividad {sel}% ---")
        ex = exact.get((sc, sel))
        print(f"{'Motor / Config':<28} | {'Recall@10':<9} | {'Recall@100':<10} | {'p50 (us)':<8} | {'p95 (us)':<8} | {'QPS':<8} | {'Leaks'}")
        print("-" * 95)
        if ex:
            print(f"{'ExactFilteredScan (Mod D)':<28} | {'100.00%':<9} | {'100.00%':<10} | {ex['p50']:8.1f} | {ex['p95']:8.1f} | {ex['qps']:8.1f} | 0")
        
        a_t1a = alg.get((sc, sel, "ALG_H2_AT_T1A"))
        if a_t1a:
            print(f"{'ALG H2-AT T1-A (512)':<28} | {a_t1a['R10']*100:6.2f}%   | {a_t1a['R100']*100:6.2f}%    | {a_t1a['p50']:8.1f} | {a_t1a['p95']:8.1f} | {a_t1a['qps']:8.1f} | {a_t1a['leaks']}")

        a_base = alg.get((sc, sel, "ALG_H2_Base"))
        if a_base:
            print(f"{'ALG H2-Base (ef=100)':<28} | {a_base['R10']*100:6.2f}%   | {a_base['R100']*100:6.2f}%    | {a_base['p50']:8.1f} | {a_base['p95']:8.1f} | {a_base['qps']:8.1f} | {a_base['leaks']}")

        for ef in [50, 100, 200, 500]:
            h = hnsw.get((sc, sel, ef))
            if h:
                name = f"hnswlib (ef={ef})"
                print(f"{name:<28} | {h['R10']*100:6.2f}%   | {h['R100']*100:6.2f}%    | {h['p50']:8.1f} | {h['p95']:8.1f} | {h['qps']:8.1f} | {h['leaks']}")
