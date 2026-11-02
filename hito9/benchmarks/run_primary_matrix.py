import os
import subprocess
import time

cases = [
    {"mode": "R", "basins": 64, "sigma": 0.03, "fname": "ood_seed8_k64_s003_r.bin"},
    {"mode": "R", "basins": 64, "sigma": 0.25, "fname": "ood_seed8_k64_s025_r.bin"},
    {"mode": "D", "basins": 64, "sigma": 0.09, "fname": "ood_seed8_k64_s009_d.bin"},
    {"mode": "R", "basins": 16, "sigma": 0.09, "fname": "ood_seed8_k16_s009_r.bin"},
    {"mode": "R", "basins": 128, "sigma": 0.09, "fname": "ood_seed8_k128_s009_r.bin"},
]

manifests_dir = os.path.join("hito9", "datasets", "manifests")
os.makedirs(manifests_dir, exist_ok=True)

for c in cases:
    bin_path = os.path.join(manifests_dir, c["fname"])
    print(f"\n========================================================")
    print(f"GENERANDO: {c['fname']} (Mode={c['mode']}, K={c['basins']}, sigma={c['sigma']})")
    print(f"========================================================")
    cmd_gen = [
        r".\hito9\datasets\generate_ood.exe",
        "--seed", "8",
        "--basins", str(c["basins"]),
        "--sigma", str(c["sigma"]),
        "--mix", c["mode"],
        "--out", bin_path
    ]
    subprocess.run(cmd_gen, check=True)

    print(f"\nEJECUTANDO BENCHMARK RUNNER SOBRE {c['fname']}...")
    t0 = time.perf_counter()
    subprocess.run([r".\hito9\benchmarks\run_hito9.exe", bin_path], check=True)
    t1 = time.perf_counter()
    print(f"Completado {c['fname']} en {t1 - t0:.2f} s")
    time.sleep(1)

print("\nTodos los casos de la Matriz Primaria OOD fueron completados exitosamente.")
