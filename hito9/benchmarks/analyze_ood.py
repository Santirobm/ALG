import os
import glob
import pandas as pd
import numpy as np

def main():
    summary_files = glob.glob("hito9/results/ood_*.bin_summary.csv")
    if not summary_files:
        print("No se encontraron archivos de resultados OOD.")
        return

    df = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
    print("========================================================================")
    print("HITO 9: EVALUACION EXPERIMENTAL OOD (TRACK 2 - DISTRIBUCIONES FUERA DE MUESTRA)")
    print("MATRIZ PRIMARIA: MEZCLAS M50/50, DISPERSIONES (0.03, 0.09, 0.25) Y CUENCAS (16, 64, 128)")
    print("========================================================================")

    # Resumen por Caso OOD y Estrategia
    cases = df.groupby(['MixMode', 'VectorSigma', 'K_Basins'])

    for (mix, sigma, k_basins), sub in cases:
        print(f"\n========================================================================")
        print(f"CASO OOD: Mix={mix} | Sigma={sigma} | Cuencas K={k_basins}")
        print(f"========================================================================")

        strat_grp = sub.groupby('Strategy').mean(numeric_only=True).reset_index()
        print(f"{'Estrategia':<24} | {'p50 (us)':<10} | {'p95 (us)':<10} | {'R@10 (%)':<10} | {'R@100 (%)':<10} | {'% Exact':<8} | {'% HNSW':<8} | {'+Regret Med':<12} | {'+Regret p95':<12}")
        print("-" * 120)
        for _, r in strat_grp.sort_values(by='p50_us').iterrows():
            print(f"{r['Strategy']:<24} | {r['p50_us']:<10.1f} | {r['p95_us']:<10.1f} | {r['Recall10']:<10.2f} | {r['Recall100']:<10.2f} | {r['PctExact']:<8.1f} | {r['PctHnsw']:<8.1f} | {r['PositiveRegretMean']:<12.2f} | {r['PositiveRegret_p95']:<12.2f}")

    # Tabla comparativa general de Robustez
    print(f"\n========================================================================")
    print("SINTESIS GLOBAL DE ROBUSTEZ: PROMEDIO EN TODOS LOS ESCENARIOS OOD")
    print("========================================================================")
    global_strat = df.groupby('Strategy').mean(numeric_only=True).reset_index()
    print(f"{'Estrategia':<24} | {'Latencia Med':<14} | {'R@10 (%)':<10} | {'R@100 (%)':<10} | {'% Exact':<8} | {'% HNSW':<8} | {'+Regret Med (us)':<18} | {'+Regret p95 (us)':<18}")
    print("-" * 120)
    for _, r in global_strat.sort_values(by='PositiveRegretMean').iterrows():
        print(f"{r['Strategy']:<24} | {r['MeanLatency_us']:<14.1f} | {r['Recall10']:<10.2f} | {r['Recall100']:<10.2f} | {r['PctExact']:<8.1f} | {r['PctHnsw']:<8.1f} | {r['PositiveRegretMean']:<18.2f} | {r['PositiveRegret_p95']:<18.2f}")

    out_csv = "hito9/results/ood_primary_matrix_synthesis.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSintesis completa exportada a {out_csv}")

if __name__ == "__main__":
    main()
