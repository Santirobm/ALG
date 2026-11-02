import os
import subprocess
import time
import sys

SEEDS = [8, 9, 10]

SCENARIOS = [
    {"name": "k64_s009_m5050", "mix": "M50/50", "basins": 64, "sigma": 0.09},
    {"name": "k64_s003_r",     "mix": "R",      "basins": 64, "sigma": 0.03},
    {"name": "k64_s025_r",     "mix": "R",      "basins": 64, "sigma": 0.25},
    {"name": "k64_s009_d",     "mix": "D",      "basins": 64, "sigma": 0.09},
    {"name": "k16_s009_r",     "mix": "R",      "basins": 16, "sigma": 0.09},
    {"name": "k128_s009_r",    "mix": "R",      "basins": 128, "sigma": 0.09},
]

manifests_dir = os.path.join("hito9", "datasets", "manifests")
os.makedirs(manifests_dir, exist_ok=True)
results_dir = os.path.join("hito9", "results")
os.makedirs(results_dir, exist_ok=True)

gen_exe = os.path.join("hito9", "datasets", "generate_ood.exe")
bench_exe = os.path.join("hito9", "benchmarks", "run_hito9.exe")

print(f"========================================================================")
print(f"HITO 9: EJECUCION DE BATERIA MULTI-SEMILLA FACTORIAL (SEEDS 8, 9, 10)")
print(f"6 Escenarios OOD x 3 Semillas = 18 Datasets Experimentales")
print(f"========================================================================")

total_start = time.perf_counter()

for s in SEEDS:
    print(f"\n########################################################################")
    print(f"### INICIANDO MATRIZ COMPLETA PARA SEMILLA {s}")
    print(f"########################################################################")
    
    for sc in SCENARIOS:
        fname = f"ood_seed{s}_{sc['name']}.bin"
        bin_path = os.path.join(manifests_dir, fname)
        
        # Generar dataset si no existe
        if not os.path.exists(bin_path):
            print(f"\n[GENERACION] Seed={s} | Escenario={sc['name']} -> {fname}")
            cmd_gen = [
                gen_exe,
                "--seed", str(s),
                "--basins", str(sc["basins"]),
                "--sigma", str(sc["sigma"]),
                "--mix", sc["mix"],
                "--out", bin_path
            ]
            subprocess.run(cmd_gen, check=True)
        else:
            print(f"\n[DATASET EXISTENTE] {fname}")
            
        print(f"[BENCHMARK] Ejecutando run_hito9.exe sobre {fname}...")
        t0 = time.perf_counter()
        subprocess.run([bench_exe, bin_path], check=True)
        t1 = time.perf_counter()
        print(f"[OK] {fname} finalizado en {t1 - t0:.2f} s")
        time.sleep(0.5)

total_elapsed = time.perf_counter() - total_start
print(f"\n========================================================================")
print(f"BATERIA DE 18 BENCHMARKS COMPLETADA EXITOSAMENTE EN {total_elapsed/60:.2f} MINUTOS")
print(f"========================================================================")

# Ejecutar sintesis OOD
print("\nEjecutando sintesis global OOD (analyze_ood.py)...")
subprocess.run([sys.executable, os.path.join("hito9", "benchmarks", "analyze_ood.py")], check=True)

# Ejecutar Bootstrap jerarquico bi-nivel
print("\nEjecutando Bootstrap Jerarquico Bi-Nivel (bootstrap_regret.py)...")
subprocess.run([sys.executable, os.path.join("hito9", "benchmarks", "bootstrap_regret.py")], check=True)

print("\nTodo el pipeline multi-semilla ha culminado con exito.")
