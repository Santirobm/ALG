import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import struct
import ctypes
import ctypes.wintypes
import numpy as np
import faiss
faiss.omp_set_num_threads(1)

# --- Windows Native Setup (Process Affinity & Priority) ---
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

HIGH_PRIORITY_CLASS = 0x00000080
kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)
kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1)
kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), 1)

class PMC(ctypes.Structure):
    _fields_ = [
        ('cb', ctypes.c_ulong),
        ('PageFaultCount', ctypes.c_ulong),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]

psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
psapi.GetProcessMemoryInfo.restype = ctypes.c_int

def get_working_set_bytes():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return pmc.WorkingSetSize
    return 0

def recall_at_k(pred_ids, gt_ids, gt_count, k):
    n_gt = min(gt_count, k)
    if n_gt == 0:
        return 1.0
    # Filter out -1 invalid IDs
    valid_pred = [x for x in pred_ids[:k] if x >= 0]
    if not valid_pred:
        return 0.0
    gt_set = set(gt_ids[:n_gt])
    match = sum(1 for x in valid_pred if x in gt_set)
    return match / float(n_gt)

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else "hito7/data/dataset_seed1.bin"

    print("========================================================================")
    print("Hito 6: Adaptador FAISS (IndexHNSWFlat + IDSelector) sobre Dataset Comun")
    print(f"Archivo: {data_file}")
    print("========================================================================")

    if not os.path.exists(data_file):
        print(f"Error: No existe el archivo {data_file}")
        sys.exit(1)

    with open(data_file, "rb") as f:
        fixed = f.read(36)
        if len(fixed) != 36:
            raise RuntimeError("Header truncado")
        magic, ver, s_idx, s_val, nv, dim, nq, n_sel = struct.unpack('<IIIQIIII', fixed)
        hdr_bytes = fixed + f.read(n_sel * 8)
        if magic != 0x414C4736 or ver != 2:
            print("Error: Magic header invalido")
            sys.exit(2)
        if len(hdr_bytes) != 36 + n_sel * 8:
            raise RuntimeError("Header v2 incompleto")
        selectivities = struct.unpack(f'<{n_sel}I', hdr_bytes[36:36 + n_sel * 4])
        target_categories = struct.unpack(f'<{n_sel}I', hdr_bytes[36 + n_sel * 4:36 + n_sel * 8])

        print(f"Cargando datos: {nv} vectores (dim={dim}), {nq} consultas...")
        vectors = np.fromfile(f, dtype=np.float32, count=nv * dim).reshape((nv, dim))
        categories = np.fromfile(f, dtype=np.uint32, count=nv)
        queries_R = np.fromfile(f, dtype=np.float32, count=nq * dim).reshape((nq, dim))
        queries_D = np.fromfile(f, dtype=np.float32, count=nq * dim).reshape((nq, dim))

        # Ground truth
        TOP_K_GT = 100
        # gt[sc][seli] -> list of (count, ids_array, dists_array)
        gt_data = [[[] for _ in range(n_sel)] for _ in range(2)]
        for sc in range(2):
            for seli in range(n_sel):
                for q in range(nq):
                    ret_count = struct.unpack('<I', f.read(4))[0]
                    gt_ids = np.fromfile(f, dtype=np.uint32, count=TOP_K_GT)
                    gt_dists = np.fromfile(f, dtype=np.float32, count=TOP_K_GT)
                    gt_data[sc][seli].append((ret_count, gt_ids, gt_dists))

    print("Datos y Ground Truth cargados correctamente.")

    # Build IndexHNSWFlat
    mem_before = get_working_set_bytes()
    print("\nConstruyendo indice FAISS IndexHNSWFlat (M=16, efConstruction=64)...")
    t0 = time.perf_counter()

    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 64
    index.add(vectors)

    build_sec = time.perf_counter() - t0
    mem_after = get_working_set_bytes()
    index_mem_mb = (mem_after - mem_before) / (1024.0 * 1024.0)

    print(f"FAISS construido en {build_sec:.3f} s | Memoria residente adicional: {index_mem_mb:.2f} MiB")

    os.makedirs("hito7/results", exist_ok=True)
    f_summary = open(f"hito7/results/faiss_seed{s_idx}_summary.csv", "w", encoding="utf-8")
    f_summary.write("Seed,Scenario,Selectivity,Variant,efSearch,Recall10,Recall100,p50_us,p95_us,p99_us,QPS,BuildSec,IndexMemMiB,Leaks\n")

    f_per_query = open(f"hito7/results/faiss_seed{s_idx}_per_query.csv", "w", encoding="utf-8")
    f_per_query.write("Seed,Scenario,Selectivity,Variant,efSearch,QueryID,Latency_us,Recall10,Recall100,ReturnedCount,Leaks\n")

    ef_search_values = [50, 100, 150, 200, 300, 500]

    print("\n--- Ejecutando evaluacion experimental de FAISS con SearchParametersHNSW + IDSelector ---")

    for sc in range(2):
        scenario_name = "R_Local" if sc == 0 else "D_Adversarial"
        q_matrix = queries_R if sc == 0 else queries_D
        print(f"\n==================== Escenario {scenario_name} ====================")

        for seli in range(n_sel):
            sel = selectivities[seli]
            target_cat = target_categories[seli]

            eligible_mask = (categories < target_cat)
            sel_label = f"{sel / 10.0:.1f}"
            eligible_ids = np.where(eligible_mask)[0].astype(np.int64)

            # Build IDSelectorBatch
            id_selector = faiss.IDSelectorBatch(eligible_ids)
            print(f"\n-- Selectividad {sel_label}% (Categoria objetivo={target_cat} | Elegibles={len(eligible_ids)}) --")

            for ef in ef_search_values:
                params = faiss.SearchParametersHNSW()
                params.efSearch = ef
                params.sel = id_selector

                # Warmup
                for w in range(5):
                    qw = q_matrix[w:w+1]
                    index.search(qw, 100, params=params)

                latencies = []
                sum_r10 = 0.0
                sum_r100 = 0.0
                total_leaks = 0

                t_batch_0 = time.perf_counter()
                for q in range(nq):
                    query_vec = q_matrix[q:q+1]
                    
                    qa = time.perf_counter_ns()
                    D, I = index.search(query_vec, 100, params=params)
                    qb = time.perf_counter_ns()

                    us = (qb - qa) / 1000.0
                    latencies.append(us)

                    res_ids = I[0]
                    query_leaks = 0
                    valid_count = 0
                    for rid in res_ids:
                        if rid < 0:
                            continue
                        valid_count += 1
                        if not eligible_mask[rid]:
                            query_leaks += 1
                            total_leaks += 1

                    gt_count, gt_ids, _ = gt_data[sc][seli][q]
                    r10 = recall_at_k(res_ids, gt_ids, gt_count, 10)
                    r100 = recall_at_k(res_ids, gt_ids, gt_count, 100)
                    sum_r10 += r10
                    sum_r100 += r100

                    f_per_query.write(f"{s_idx},{scenario_name},{sel_label},FAISS,{ef},{q},{us:.2f},{r10:.4f},{r100:.4f},{valid_count},{query_leaks}\n")

                t_batch_1 = time.perf_counter()
                total_sec = t_batch_1 - t_batch_0
                qps = nq / total_sec if total_sec > 0 else 0.0

                latencies.sort()
                p50 = latencies[int(nq * 0.50)]
                p95 = latencies[int(nq * 0.95)]
                p99 = latencies[int(nq * 0.99)]
                mean_r10 = sum_r10 / nq
                mean_r100 = sum_r100 / nq

                print(f"ef={ef:3d} | R@10: {mean_r10*100:6.2f}% | R@100: {mean_r100*100:6.2f}% | "
                      f"p50: {p50:6.1f} us | p95: {p95:6.1f} us | QPS: {qps:6.1f} | leaks: {total_leaks}")

                f_summary.write(f"{s_idx},{scenario_name},{sel_label},FAISS,{ef},{mean_r10:.6f},{mean_r100:.6f},"
                                f"{p50:.2f},{p95:.2f},{p99:.2f},{qps:.2f},{build_sec:.3f},{index_mem_mb:.2f},{total_leaks}\n")
                f_summary.flush()
                f_per_query.flush()

    f_summary.close()
    f_per_query.close()
    print("\nEvaluacion de FAISS completada exitosamente.")

if __name__ == "__main__":
    main()
