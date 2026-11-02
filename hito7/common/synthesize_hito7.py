import os
import glob
import pandas as pd
import numpy as np

def load_all_summaries():
    exact_files = glob.glob("hito7/results/exact_scan_seed*_summary.csv")
    hnsw_files = glob.glob("hito7/results/hnswlib_seed*_summary.csv")
    faiss_files = glob.glob("hito7/results/faiss_seed*_summary.csv")
    alg_files = glob.glob("hito7/results/alg_seed*_summary.csv")
    topo_files = glob.glob("hito7/results/topology_seed*_summary.csv")

    df_exact = pd.concat([pd.read_csv(f) for f in exact_files], ignore_index=True)
    df_hnsw = pd.concat([pd.read_csv(f) for f in hnsw_files], ignore_index=True)
    df_faiss = pd.concat([pd.read_csv(f) for f in faiss_files], ignore_index=True)
    df_alg = pd.concat([pd.read_csv(f) for f in alg_files], ignore_index=True)
    df_topo = pd.concat([pd.read_csv(f) for f in topo_files], ignore_index=True)

    return df_exact, df_hnsw, df_faiss, df_alg, df_topo

def main():
    df_exact, df_hnsw, df_faiss, df_alg, df_topo = load_all_summaries()

    print("========================================================================")
    print("HITO 7: SINTESIS ESTADISTICA DE 5 SEMILLAS (CROSSOVER Y PERCOLACION)")
    print("========================================================================")

    # 1. Total leaks check
    total_leaks = df_hnsw['Leaks'].sum() + df_faiss['Leaks'].sum() + df_alg['Leaks'].sum()
    print(f"\n[1] VERIFICACION DE INTEGRIDAD DE FILTRADO:")
    print(f"Total de violaciones de filtro (Leaks) detectadas en todas las semillas y motores: {total_leaks}")

    selectivities = [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]

    # 2. Crossover per seed for Scenario R and D
    print(f"\n[2] CROSSOVER DE LATENCIA CRUDA POR SEMILLA (ExactScan vs hnswlib ef=50):")
    for sc in ['R_Local', 'D_Adversarial']:
        print(f"\n--- Escenario: {sc} ---")
        for s in range(1, 6):
            co_point = None
            for sel in selectivities:
                e_p50 = df_exact[(df_exact.Seed == s) & (df_exact.Scenario == sc) & (df_exact.Selectivity == sel)]['p50_us'].values[0]
                h_p50 = df_hnsw[(df_hnsw.Seed == s) & (df_hnsw.Scenario == sc) & (df_hnsw.Selectivity == sel) & (df_hnsw.efSearch == 50)]['p50_us'].values[0]
                if h_p50 < e_p50 and co_point is None:
                    co_point = sel
            print(f"Semilla {s}: Crossover en {co_point}% (ExactScan lidera en <= {co_point - 2.5}%, hnswlib lidera en >= {co_point}%)")

    # 3. Aggregated Mean and Std across 5 seeds for Scenario R
    print(f"\n[3] MATRIZ AGREGADA MULTI-SEMILLA (Escenario R_Local, Media +/- Desv. Estandar):")
    print(f"{'Selectividad':<12} | {'ExactScan p50 (us)':<22} | {'hnswlib ef=50 p50':<22} | {'hnswlib ef=50 R@10':<20} | {'FAISS ef=50 p50':<20} | {'FAISS ef=50 R@10':<18}")
    print("-" * 125)

    summary_rows = []
    for sel in selectivities:
        e_sub = df_exact[(df_exact.Scenario == 'R_Local') & (df_exact.Selectivity == sel)]['p50_us']
        h_sub = df_hnsw[(df_hnsw.Scenario == 'R_Local') & (df_hnsw.Selectivity == sel) & (df_hnsw.efSearch == 50)]
        f_sub = df_faiss[(df_faiss.Scenario == 'R_Local') & (df_faiss.Selectivity == sel) & (df_faiss.efSearch == 50)]

        e_mean, e_std = e_sub.mean(), e_sub.std()
        h_mean, h_std = h_sub['p50_us'].mean(), h_sub['p50_us'].std()
        hr_mean, hr_std = h_sub['Recall10'].mean() * 100, h_sub['Recall10'].std() * 100
        f_mean, f_std = f_sub['p50_us'].mean(), f_sub['p50_us'].std()
        fr_mean, fr_std = f_sub['Recall10'].mean() * 100, f_sub['Recall10'].std() * 100

        print(f"{sel:5.1f}%       | {e_mean:6.1f} +/- {e_std:4.1f} us        | {h_mean:6.1f} +/- {h_std:4.1f} us        | {hr_mean:5.2f}% +/- {hr_std:4.2f}%     | {f_mean:5.1f} +/- {f_std:4.1f} us      | {fr_mean:5.2f}% +/- {fr_std:4.2f}%")

        summary_rows.append({
            "Scenario": "R_Local",
            "Selectivity": sel,
            "Exact_p50_mean": e_mean, "Exact_p50_std": e_std,
            "hnsw_ef50_p50_mean": h_mean, "hnsw_ef50_p50_std": h_std,
            "hnsw_ef50_R10_mean": hr_mean, "hnsw_ef50_R10_std": hr_std,
            "faiss_ef50_p50_mean": f_mean, "faiss_ef50_p50_std": f_std,
            "faiss_ef50_R10_mean": fr_mean, "faiss_ef50_R10_std": fr_std
        })

    # 4. Aggregated Mean and Std across 5 seeds for Scenario D
    print(f"\n[4] MATRIZ AGREGADA MULTI-SEMILLA (Escenario D_Adversarial, Media +/- Desv. Estandar):")
    print(f"{'Selectividad':<12} | {'ExactScan p50 (us)':<22} | {'hnswlib ef=50 p50':<22} | {'hnswlib ef=50 R@10':<20} | {'FAISS ef=50 p50':<20} | {'FAISS ef=50 R@10':<18}")
    print("-" * 125)

    for sel in selectivities:
        e_sub = df_exact[(df_exact.Scenario == 'D_Adversarial') & (df_exact.Selectivity == sel)]['p50_us']
        h_sub = df_hnsw[(df_hnsw.Scenario == 'D_Adversarial') & (df_hnsw.Selectivity == sel) & (df_hnsw.efSearch == 50)]
        f_sub = df_faiss[(df_faiss.Scenario == 'D_Adversarial') & (df_faiss.Selectivity == sel) & (df_faiss.efSearch == 50)]

        e_mean, e_std = e_sub.mean(), e_sub.std()
        h_mean, h_std = h_sub['p50_us'].mean(), h_sub['p50_us'].std()
        hr_mean, hr_std = h_sub['Recall10'].mean() * 100, h_sub['Recall10'].std() * 100
        f_mean, f_std = f_sub['p50_us'].mean(), f_sub['p50_us'].std()
        fr_mean, fr_std = f_sub['Recall10'].mean() * 100, f_sub['Recall10'].std() * 100

        print(f"{sel:5.1f}%       | {e_mean:6.1f} +/- {e_std:4.1f} us        | {h_mean:6.1f} +/- {h_std:4.1f} us        | {hr_mean:5.2f}% +/- {hr_std:4.2f}%     | {f_mean:5.1f} +/- {f_std:4.1f} us      | {fr_mean:5.2f}% +/- {fr_std:4.2f}%")

    # 5. Iso-Recall Crossover (Latencia cuando exigimos Recall >= 95% o >= 98%)
    print(f"\n[5] COMPARACION A ISO-RECALL (Escenario R_Local, Exigiendo Recall@10 >= 95%):")
    for sel in selectivities:
        e_p50 = df_exact[(df_exact.Scenario == 'R_Local') & (df_exact.Selectivity == sel)]['p50_us'].mean()
        # Find minimum efSearch where mean Recall10 >= 0.95
        h_valid = df_hnsw[(df_hnsw.Scenario == 'R_Local') & (df_hnsw.Selectivity == sel)].groupby('efSearch').mean(numeric_only=True)
        h_qual = h_valid[h_valid['Recall10'] >= 0.95]
        if not h_qual.empty:
            min_ef = h_qual.index.min()
            h_p50 = h_qual.loc[min_ef, 'p50_us']
            h_rec = h_qual.loc[min_ef, 'Recall10'] * 100
            print(f"Selectividad {sel:4.1f}% | ExactScan (100% Rec): {e_p50:5.1f} us | hnswlib ef={min_ef:<3d} ({h_rec:5.2f}% Rec): {h_p50:5.1f} us -> Ganador: {'ExactScan' if e_p50 < h_p50 else 'hnswlib'}")
        else:
            print(f"Selectividad {sel:4.1f}% | ExactScan (100% Rec): {e_p50:5.1f} us | hnswlib no alcanza 95% -> Ganador: ExactScan")

    # 6. Topological Percolation across all seeds
    print(f"\n[6] PROMEDIO DE TOPOLOGIA DEL SUBGRAFO INDUCIDO G[V_el] (5 Semillas):")
    t_grp = df_topo.groupby('Selectivity').mean(numeric_only=True)
    print(f"{'Selectividad':<12} | {'Elegibles':<10} | {'Componentes':<12} | {'Comp Gigante %':<16} | {'Singletons':<12} | {'Comps en Top10':<15} | {'GT Aislado %':<14}")
    print("-" * 105)
    for sel, row in t_grp.iterrows():
        print(f"{sel:5.1f}%       | {int(row['EligibleCount']):<10d} | {row['NumComponents']:<12.1f} | {row['GiantCompPct']:<16.2f}% | {row['Singletons']:<12.1f} | {row['AvgCompsTop10']:<15.2f} | {row['GtIsolatedFromGiantPct']:<14.2f}%")

    out_csv = "hito7/results/hito7_multi_seed_synthesis.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nSintesis exportada a {out_csv}")

if __name__ == "__main__":
    main()
