import sys
import struct
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import faiss

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else "hito7/data/dataset_seed1.bin"
    print("========================================================================")
    print("Hito 7: Instrumentacion Topologica — Componentes Conexas del Subgrafo Inducido")
    print(f"Archivo: {data_file}")
    print("========================================================================")

    with open(data_file, "rb") as f:
        fixed = f.read(36)
        magic, ver, s_idx, s_val, nv, dim, nq, n_sel = struct.unpack('<IIIQIIII', fixed)
        hdr_bytes = fixed + f.read(n_sel * 8)
        selectivities = struct.unpack(f'<{n_sel}I', hdr_bytes[36:36 + n_sel * 4])
        target_categories = struct.unpack(f'<{n_sel}I', hdr_bytes[36 + n_sel * 4:36 + n_sel * 8])

        print(f"Cargando {nv} vectores (dim={dim}) y categorias...")
        vectors = np.fromfile(f, dtype=np.float32, count=nv * dim).reshape((nv, dim))
        categories = np.fromfile(f, dtype=np.uint32, count=nv)
        queries_R = np.fromfile(f, dtype=np.float32, count=nq * dim).reshape((nq, dim))
        queries_D = np.fromfile(f, dtype=np.float32, count=nq * dim).reshape((nq, dim))

        # Ground truth top-10
        TOP_K_GT = 100
        gt_data = [[[] for _ in range(n_sel)] for _ in range(2)]
        for sc in range(2):
            for seli in range(n_sel):
                for q in range(nq):
                    ret_count = struct.unpack('<I', f.read(4))[0]
                    gt_ids = np.fromfile(f, dtype=np.uint32, count=TOP_K_GT)
                    gt_dists = np.fromfile(f, dtype=np.float32, count=TOP_K_GT)
                    gt_data[sc][seli].append(gt_ids[:10])

    print("\nConstruyendo grafo HNSW de referencia (M=16, efConstruction=64)...")
    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 64
    index.add(vectors)

    nbrs = faiss.vector_to_array(index.hnsw.neighbors)
    offs = faiss.vector_to_array(index.hnsw.offsets)
    m0 = index.hnsw.nb_neighbors(0)

    print("\n================ Analisis de Topologia del Subgrafo Inducido ================")
    print("Selectivity | Elegibles | Components | Giant Comp % | Singletons | Avg Comp Top10 GT | R Isolated %")
    print("-" * 88)

    results = []

    for seli in range(n_sel):
        sel_pct = selectivities[seli] / 10.0
        target_cat = target_categories[seli]
        eligible_mask = (categories < target_cat)
        el_indices = np.where(eligible_mask)[0]
        n_el = len(el_indices)

        global_to_local = {gid: lid for lid, gid in enumerate(el_indices)}
        rows, cols = [], []
        for lid, u in enumerate(el_indices):
            base = offs[u]
            for k in range(m0):
                v = nbrs[base + k]
                if v >= 0 and v in global_to_local:
                    rows.append(lid)
                    cols.append(global_to_local[v])

        adj = sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n_el, n_el))
        n_comp, labels = connected_components(adj, directed=False)
        sizes = np.bincount(labels)
        giant_size = sizes.max()
        giant_pct = (giant_size / n_el) * 100.0
        singletons = int(np.sum(sizes == 1))

        # Check Ground Truth component dispersion for Scenario R
        gt_comp_counts = []
        gt_in_giant_count = 0
        total_gt_checked = 0
        for q in range(nq):
            gt10 = gt_data[0][seli][q]
            q_comps = set()
            for gid in gt10:
                if gid in global_to_local:
                    lid = global_to_local[gid]
                    c_id = labels[lid]
                    q_comps.add(c_id)
                    if sizes[c_id] == giant_size:
                        gt_in_giant_count += 1
                    total_gt_checked += 1
            gt_comp_counts.append(len(q_comps))

        avg_comps_top10 = np.mean(gt_comp_counts)
        pct_in_giant = (gt_in_giant_count / total_gt_checked) * 100.0 if total_gt_checked else 0.0

        print(f"{sel_pct:9.1f}% | {n_el:9d} | {n_comp:10d} | {giant_pct:11.2f}% | {singletons:10d} | {avg_comps_top10:18.2f} | {100.0 - pct_in_giant:11.2f}%")

        results.append({
            "Selectivity": sel_pct,
            "EligibleCount": n_el,
            "NumComponents": n_comp,
            "GiantCompPct": giant_pct,
            "Singletons": singletons,
            "AvgCompsTop10": avg_comps_top10,
            "GtIsolatedFromGiantPct": 100.0 - pct_in_giant
        })

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("hito7/results/topology_seed1_summary.csv", index=False)
    print("\nResultados exportados a hito7/results/topology_seed1_summary.csv")

if __name__ == "__main__":
    main()
