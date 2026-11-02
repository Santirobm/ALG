import os
import glob
import pandas as pd
import numpy as np

def load_hito8_data():
    summary_files = glob.glob("hito8/results/hito8_seed*_summary.csv")
    per_query_files = glob.glob("hito8/results/hito8_seed*_per_query.csv")

    df_summary = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True) if summary_files else pd.DataFrame()
    return df_summary

def main():
    df_summary = load_hito8_data()
    if df_summary.empty:
        print("No se encontraron resultados de Hito 8.")
        return

    print("========================================================================")
    print("HITO 8: EVALUACION EXPERIMENTAL CIEGA (OUT-OF-SAMPLE SEEDS 6 Y 7)")
    print("ANALISIS COMPARATIVO DE POLITICAS, REGRET Y ENRUTAMIENTO")
    print("========================================================================")

    seeds = sorted(df_summary['Seed'].unique())
    print(f"Semillas evaluadas: {seeds}")

    # Agrupar por Escenario, Selectividad y Estrategia
    grp = df_summary.groupby(['Scenario', 'Selectivity', 'Strategy']).mean(numeric_only=True).reset_index()

    for sc in ['R_Local', 'D_Adversarial']:
        print(f"\n========================================================================")
        print(f"ESCENARIO: {sc}")
        print(f"========================================================================")
        
        sub = grp[grp.Scenario == sc]
        selectivities = sorted(sub['Selectivity'].unique())

        for sel in selectivities:
            print(f"\n--- Selectividad: {sel:.1f}% ---")
            print(f"{'Estrategia':<24} | {'p50 (us)':<10} | {'p95 (us)':<10} | {'R@10 (%)':<10} | {'R@100 (%)':<10} | {'% Exact':<8} | {'% HNSW':<8} | {'Regret Med':<11} | {'Regret p95':<11}")
            print("-" * 115)
            
            sel_rows = sub[sub.Selectivity == sel].sort_values(by='p50_us')
            for _, r in sel_rows.iterrows():
                print(f"{r['Strategy']:<24} | {r['p50_us']:<10.1f} | {r['p95_us']:<10.1f} | {r['Recall10']:<10.2f} | {r['Recall100']:<10.2f} | {r['PctExact']:<8.1f} | {r['PctHnsw']:<8.1f} | {r['MeanRegret_us']:<11.2f} | {r['p95Regret_us']:<11.2f}")

    # Resumen general de Regret acumulado
    print(f"\n========================================================================")
    print("RESUMEN GLOBAL DE REGRET Y DESEMPENO POR ESTRATEGIA (PROMEDIO EN TODO EL ESPECTRO)")
    print("========================================================================")
    global_strat = df_summary.groupby('Strategy').mean(numeric_only=True).reset_index()
    print(f"{'Estrategia':<24} | {'Latencia Med':<14} | {'R@10 (%)':<10} | {'R@100 (%)':<10} | {'% Exact':<8} | {'% HNSW':<8} | {'Regret Med (us)':<16} | {'Regret p95 (us)':<16}")
    print("-" * 115)
    for _, r in global_strat.sort_values(by='MeanRegret_us').iterrows():
        print(f"{r['Strategy']:<24} | {r['MeanLatency_us']:<14.1f} | {r['Recall10']:<10.2f} | {r['Recall100']:<10.2f} | {r['PctExact']:<8.1f} | {r['PctHnsw']:<8.1f} | {r['MeanRegret_us']:<16.2f} | {r['p95Regret_us']:<16.2f}")

    # Exportar síntesis agregada
    out_path = "hito8/results/hito8_multi_seed_synthesis.csv"
    grp.to_csv(out_path, index=False)
    print(f"\nSintesis completa guardada en {out_path}")

if __name__ == "__main__":
    main()
