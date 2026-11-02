import sys
import os
import subprocess
import time
import argparse

def run_command(cmd_list, desc):
    print(f"\n========================================================")
    print(f"EJECUTANDO: {desc}")
    print(f"Comando: {' '.join(cmd_list)}")
    print(f"========================================================")
    t0 = time.perf_counter()
    res = subprocess.run(cmd_list, cwd=".")
    t1 = time.perf_counter()
    if res.returncode != 0:
        print(f"ERROR: {desc} fallo con codigo de salida {res.returncode}")
        sys.exit(res.returncode)
    print(f"Completado: {desc} en {t1 - t0:.2f} s")
    time.sleep(2) # Pausa de enfriamiento termico

def main():
    parser = argparse.ArgumentParser(description="Hito 7 Benchmark Runner")
    parser.add_argument("--seed-idx", type=int, default=None, help="Semilla especifica (1..5) o todas")
    parser.add_argument("--engine", type=str, default="all", choices=["all", "exact", "hnswlib", "faiss", "alg", "topo"])
    args = parser.parse_args()

    seeds = [args.seed_idx] if args.seed_idx else [1, 2, 3, 4, 5]

    for s in seeds:
        data_file = f"hito7/data/dataset_seed{s}.bin"
        if not os.path.exists(data_file):
            print(f"Error: {data_file} no existe. Genere el dataset primero.")
            continue

        print(f"\n########################################################################")
        print(f"INICIANDO EVALUACION HITO 7 — SEMILLA {s} ({data_file})")
        print(f"########################################################################")

        if args.engine in ["all", "exact"]:
            run_command([r".\hito7\exact\adapter_exact.exe", data_file], f"ExactScan (Semilla {s})")

        if args.engine in ["all", "hnswlib"]:
            run_command([r".\hito7\hnswlib\adapter_hnswlib.exe", data_file], f"hnswlib (Semilla {s})")

        if args.engine in ["all", "faiss"]:
            run_command([sys.executable, r"hito7/faiss/adapter_faiss.py", data_file], f"FAISS (Semilla {s})")

        if args.engine in ["all", "alg"]:
            run_command([r".\hito7\alg\adapter_alg.exe", data_file], f"ALG H2-AT / Base (Semilla {s})")

        if args.engine in ["all", "topo"]:
            run_command([sys.executable, r"hito7/common/analyze_topology.py", data_file], f"Topologia (Semilla {s})")

    print("\nEvaluacion completada exitosamente.")

if __name__ == "__main__":
    main()
