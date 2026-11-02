#!/usr/bin/env python3
"""
Hito 9: Track 4 — Hierarchical Bootstrap Regret & Statistical Uncertainty Analyzer
Two-level hierarchical cluster bootstrap:
  - Level 1: Resample seeds (primary sampling units) with replacement.
  - Level 2: Resample queries with replacement within each selected seed cluster.
Separates policy-conditioned regret from unconstrained diagnostic regret.
"""

import os
import glob
import numpy as np
import pandas as pd

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 4242

def hierarchical_bootstrap_ci(data_df, value_col, cluster_col='Seed', n_rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    """
    Computes true two-level hierarchical bootstrap 95% confidence intervals for the mean.
    - Level 1: resamples primary clusters (Seeds) with replacement.
    - Level 2: resamples observations (Queries) with replacement within each cluster.
    If only one cluster exists, notes single-seed conditioning.
    """
    if value_col not in data_df.columns:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    point_estimate = float(np.mean(data_df[value_col]))
    
    unique_clusters = data_df[cluster_col].unique() if cluster_col in data_df.columns else []
    n_clusters = len(unique_clusters)

    if n_clusters <= 1:
        # Fallback query-level bootstrap conditioned on single seed
        values = data_df[value_col].to_numpy()
        n = len(values)
        if n == 0:
            return np.nan, np.nan, np.nan
        indices = rng.integers(0, n, size=(n_rounds, n))
        boot_means = np.mean(values[indices], axis=1)
    else:
        # True two-level cluster bootstrap
        cluster_arrays = {c: data_df[data_df[cluster_col] == c][value_col].to_numpy() for c in unique_clusters}
        boot_means = np.empty(n_rounds, dtype=np.float64)
        for r in range(n_rounds):
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
            sampled_chunks = []
            for sc in sampled_clusters:
                c_vals = cluster_arrays[sc]
                c_len = len(c_vals)
                c_idx = rng.integers(0, c_len, size=c_len)
                sampled_chunks.append(c_vals[c_idx])
            boot_means[r] = np.mean(np.concatenate(sampled_chunks))

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return point_estimate, ci_low, ci_high

def analyze_all_results():
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    pattern = os.path.join(results_dir, '*_per_query.csv')
    files = glob.glob(pattern)

    if not files:
        print(f"No per-query CSV files found matching {pattern}")
        return

    print(f"Found {len(files)} per-query CSV files. Loading data...")
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total query evaluations loaded: {len(full_df):,}")

    unique_seeds = full_df['Seed'].unique() if 'Seed' in full_df.columns else []
    print(f"Primary cluster units (Seeds) detected: {list(unique_seeds)} (Count = {len(unique_seeds)})")
    if len(unique_seeds) == 1:
        print("  [AVISO METODOLOGICO] Datos con semilla unica: el remuestreo mide variabilidad condicionada a dicha semilla.")
    else:
        print("  [METODOLOGIA] Remuestreo jerarquico activo en 2 niveles (Nivel 1: Semillas, Nivel 2: Consultas).")

    # Column resolution for backward/forward compatibility
    pol_reg_col = 'PolicyPositiveRegret_us' if 'PolicyPositiveRegret_us' in full_df.columns else 'PositiveRegret_us'
    diag_reg_col = 'DiagnosticPositiveRegret_us' if 'DiagnosticPositiveRegret_us' in full_df.columns else 'PositiveRegret_us'

    records = []
    groups = full_df.groupby(['Dataset', 'Selectivity', 'Strategy'], sort=True)
    total_groups = len(groups)
    print(f"Computing hierarchical bootstrap over {total_groups} strata (B={BOOTSTRAP_ROUNDS}, seed={BOOTSTRAP_SEED})...")

    group_idx = 0
    for (dataset, sel, strat), group in groups:
        group_idx += 1
        if group_idx % 25 == 0 or group_idx == total_groups:
            print(f"  Processing stratum {group_idx}/{total_groups}...")

        # Policy Positive Regret
        pr_mean, pr_ci_low, pr_ci_high = hierarchical_bootstrap_ci(group, pol_reg_col, cluster_col='Seed')
        pr_vals = group[pol_reg_col].to_numpy()
        pr_p50 = float(np.percentile(pr_vals, 50))
        pr_p95 = float(np.percentile(pr_vals, 95))
        pr_p99 = float(np.percentile(pr_vals, 99))

        # Diagnostic Regret
        dr_mean, dr_ci_low, dr_ci_high = hierarchical_bootstrap_ci(group, diag_reg_col, cluster_col='Seed')

        # Latency
        lat_mean, lat_ci_low, lat_ci_high = hierarchical_bootstrap_ci(group, 'Latency_us', cluster_col='Seed')
        lat_vals = group['Latency_us'].to_numpy()
        lat_p50 = float(np.percentile(lat_vals, 50))
        lat_p95 = float(np.percentile(lat_vals, 95))
        lat_p99 = float(np.percentile(lat_vals, 99))

        # Recall@10
        r10_mean, r10_ci_low, r10_ci_high = hierarchical_bootstrap_ci(group, 'Recall10', cluster_col='Seed')
        # Recall@100
        r100_mean, r100_ci_low, r100_ci_high = hierarchical_bootstrap_ci(group, 'Recall100', cluster_col='Seed')

        # Fallback fraction & Exact engine fraction
        fallback_pct = float(group['FallbackTriggered'].mean() * 100.0)
        exact_pct = float((group['ChosenEngine'] == 'EXACT').mean() * 100.0)

        records.append({
            'Dataset': dataset,
            'Selectivity_pct': sel,
            'Strategy': strat,
            'Query_Count': len(group),
            'Latency_Mean_us': round(lat_mean, 2),
            'Latency_CI95_Low': round(lat_ci_low, 2),
            'Latency_CI95_High': round(lat_ci_high, 2),
            'Latency_p50_us': round(lat_p50, 2),
            'Latency_p95_us': round(lat_p95, 2),
            'Latency_p99_us': round(lat_p99, 2),
            'Recall10_Mean': round(r10_mean, 4),
            'Recall10_CI95_Low': round(r10_ci_low, 4),
            'Recall10_CI95_High': round(r10_ci_high, 4),
            'Recall100_Mean': round(r100_mean, 4),
            'Recall100_CI95_Low': round(r100_ci_low, 4),
            'Recall100_CI95_High': round(r100_ci_high, 4),
            'PolicyRegret_Mean_us': round(pr_mean, 2),
            'PolicyRegret_CI95_Low': round(pr_ci_low, 2),
            'PolicyRegret_CI95_High': round(pr_ci_high, 2),
            'PolicyRegret_p50_us': round(pr_p50, 2),
            'PolicyRegret_p95_us': round(pr_p95, 2),
            'PolicyRegret_p99_us': round(pr_p99, 2),
            'DiagRegret_Mean_us': round(dr_mean, 2),
            'Exact_Route_pct': round(exact_pct, 1),
            'Fallback_Trigger_pct': round(fallback_pct, 1)
        })

    summary_df = pd.DataFrame(records)
    out_csv = os.path.join(results_dir, 'bootstrap_statistical_synthesis.csv')
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved statistical bootstrap synthesis to: {out_csv}")

    print("\n" + "="*105)
    print("RESUMEN GLOBAL AUDITADO POR ESTRATEGIA (CON INTERVALOS BOOTSTRAP 95%)")
    print("="*105)
    agg_records = []
    for strat, strat_group in full_df.groupby('Strategy'):
        lat_mean, lat_l, lat_h = hierarchical_bootstrap_ci(strat_group, 'Latency_us', cluster_col='Seed')
        r10_mean, r10_l, r10_h = hierarchical_bootstrap_ci(strat_group, 'Recall10', cluster_col='Seed')
        r100_mean, r100_l, r100_h = hierarchical_bootstrap_ci(strat_group, 'Recall100', cluster_col='Seed')
        pr_mean, pr_l, pr_h = hierarchical_bootstrap_ci(strat_group, pol_reg_col, cluster_col='Seed')
        dr_mean, dr_l, dr_h = hierarchical_bootstrap_ci(strat_group, diag_reg_col, cluster_col='Seed')
        pr_vals = strat_group[pol_reg_col].to_numpy()
        exact_pct = float((strat_group['ChosenEngine'] == 'EXACT').mean() * 100.0)

        agg_records.append({
            'Strategy': strat,
            'Latency_Mean_us': f"{lat_mean:.1f} [{lat_l:.1f}, {lat_h:.1f}]",
            'Recall10_Mean': f"{r10_mean*100:.2f}% [{r10_l*100:.2f}%, {r10_h*100:.2f}%]",
            'Recall100_Mean': f"{r100_mean*100:.2f}% [{r100_l*100:.2f}%, {r100_h*100:.2f}%]",
            'PolicyRegret_Mean_us': f"{pr_mean:.2f} [{pr_l:.2f}, {pr_h:.2f}]",
            'PolicyRegret_p95_us': f"{np.percentile(pr_vals, 95):.1f}",
            'DiagRegret_Mean_us': f"{dr_mean:.2f} [{dr_l:.2f}, {dr_h:.2f}]",
            'Exact_Route_pct': f"{exact_pct:.1f}%"
        })
    
    agg_df = pd.DataFrame(agg_records)
    print(agg_df.to_string(index=False))
    print("="*105)

if __name__ == '__main__':
    analyze_all_results()
