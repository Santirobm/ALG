#include "../common/dataset_format.h"
#include "../third_party/hnswlib/hnswlib/hnswlib.h"
#include <windows.h>
#include <psapi.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>

struct MemoryStats {
    size_t working_set_bytes;
    size_t private_bytes;
};

static MemoryStats get_memory_stats() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
        return { pmc.WorkingSetSize, pmc.PrivateUsage };
    }
    return { 0, 0 };
}

class CategoryFilter : public hnswlib::BaseFilterFunctor {
    const uint32_t *categories_;
    uint32_t sel_pct_;
    uint32_t target_;
public:
    CategoryFilter(const uint32_t *categories, uint32_t sel_pct, uint32_t target)
        : categories_(categories), sel_pct_(sel_pct), target_(target) {}
    bool operator()(hnswlib::labeltype label_id) override {
        if (sel_pct_ == 1U) {
            return categories_[label_id] == target_;
        }
        return categories_[label_id] < target_;
    }
};

struct QueryGT {
    uint32_t count;
    uint32_t ids[TOP_K_GT];
    float dists[TOP_K_GT];
};

static double recall_at_k(const std::vector<hnswlib::labeltype> &pred, const QueryGT &gt, uint32_t k) {
    uint32_t n_gt = gt.count < k ? gt.count : k;
    if (n_gt == 0) return 1.0;
    uint32_t n_pred = pred.size() < k ? (uint32_t)pred.size() : k;
    uint32_t match = 0;
    for (uint32_t i = 0; i < n_pred; i++) {
        for (uint32_t j = 0; j < n_gt; j++) {
            if ((uint32_t)pred[i] == gt.ids[j]) {
                match++;
                break;
            }
        }
    }
    return (double)match / (double)n_gt;
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    std::string data_file = "hito6/data/dataset_seed1.bin";
    if (argc > 1) data_file = argv[1];

    std::cout << "========================================================================\n";
    std::cout << "Hito 6: Adaptador hnswlib (C++ Native) sobre Dataset Comun\n";
    std::cout << "Archivo: " << data_file << "\n";
    std::cout << "========================================================================\n";

    FILE *f = fopen(data_file.c_str(), "rb");
    if (!f) {
        std::cerr << "Error abriendo archivo de dataset: " << data_file << "\n";
        return 1;
    }

    DatasetHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1 || header.magic != DATASET_MAGIC) {
        std::cerr << "Formato invalido en header.\n";
        fclose(f);
        return 2;
    }

    uint32_t nv = header.num_vectors;
    uint32_t dim = header.dim;
    uint32_t nq = header.num_queries;

    std::cout << "Cargando datos: " << nv << " vectores (dim=" << dim << "), " << nq << " consultas...\n";
    std::vector<float> vectors((size_t)nv * dim);
    std::vector<uint32_t> categories(nv);
    std::vector<float> queries_R((size_t)nq * dim);
    std::vector<float> queries_D((size_t)nq * dim);

    fread(vectors.data(), sizeof(float), (size_t)nv * dim, f);
    fread(categories.data(), sizeof(uint32_t), nv, f);
    fread(queries_R.data(), sizeof(float), (size_t)nq * dim, f);
    fread(queries_D.data(), sizeof(float), (size_t)nq * dim, f);

    /* Cargar Ground Truth exacto para los 10 casos */
    std::vector<QueryGT> ground_truth[2][NUM_SELECTIVITIES];
    for (int sc = 0; sc < 2; sc++) {
        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            ground_truth[sc][seli].resize(nq);
            for (uint32_t q = 0; q < nq; q++) {
                QueryGT &gt = ground_truth[sc][seli][q];
                fread(&gt.count, sizeof(uint32_t), 1, f);
                fread(gt.ids, sizeof(uint32_t), TOP_K_GT, f);
                fread(gt.dists, sizeof(float), TOP_K_GT, f);
            }
        }
    }
    fclose(f);
    std::cout << "Datos y Ground Truth cargados correctamente.\n";

    /* Construir índice hnswlib con M=16, efConstruction=64 (idéntico a ALG) */
    MemoryStats mem_before = get_memory_stats();
    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);

    std::cout << "\nConstruyendo indice hnswlib (M=16, efConstruction=64)...\n";
    QueryPerformanceCounter(&t0);
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(
        &space, nv, 16, 64, header.seed_value);

    for (size_t i = 0; i < nv; i++) {
        appr_alg->addPoint(vectors.data() + i * dim, (hnswlib::labeltype)i);
    }
    QueryPerformanceCounter(&t1);
    double build_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
    MemoryStats mem_after = get_memory_stats();

    double index_mem_mb = (double)(mem_after.working_set_bytes - mem_before.working_set_bytes) / 1048576.0;
    std::cout << "hnswlib construido en " << build_sec << " s | Memoria residente adicional: "
              << index_mem_mb << " MiB\n";

    CreateDirectoryA("hito6/results", NULL);
    std::ofstream csv_summary("hito6/results/hnswlib_summary.csv");
    csv_summary << "Seed,Scenario,Selectivity,Variant,efSearch,Recall10,Recall100,p50_us,p95_us,p99_us,QPS,BuildSec,IndexMemMiB,Leaks\n";

    std::ofstream csv_per_query("hito6/results/hnswlib_per_query.csv");
    csv_per_query << "Seed,Scenario,Selectivity,Variant,efSearch,QueryID,Latency_us,Recall10,Recall100,ReturnedCount,Leaks\n";

    /* Barrido sistemático de efSearch para trazar la curva Pareto de hnswlib */
    const std::vector<uint32_t> ef_search_values = {50, 100, 150, 200, 300, 500};

    std::cout << "\n--- Ejecutando evaluacion experimental de hnswlib con BaseFilterFunctor ---\n";

    for (int sc = 0; sc < 2; sc++) {
        const char *scenario_name = (sc == 0) ? "R_Local" : "D_Adversarial";
        const float *q_base = (sc == 0) ? queries_R.data() : queries_D.data();
        std::cout << "\n==================== Escenario " << scenario_name << " ====================\n";

        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            uint32_t sel = header.selectivities[seli];
            uint32_t target_cat = header.target_categories[seli];
            CategoryFilter filter(categories.data(), sel, target_cat);

            std::cout << "\n-- Selectividad " << sel << "% (Categoria objetivo=" << target_cat << ") --\n";

            for (uint32_t ef : ef_search_values) {
                appr_alg->setEf(ef);

                std::vector<double> latencies(nq);
                double sum_r10 = 0.0, sum_r100 = 0.0;
                uint32_t total_leaks = 0;

                QueryPerformanceCounter(&t0);
                for (uint32_t q = 0; q < nq; q++) {
                    const float *query = q_base + (size_t)q * dim;

                    LARGE_INTEGER qa, qb;
                    QueryPerformanceCounter(&qa);
                    auto result = appr_alg->searchKnn(query, 100, &filter);
                    QueryPerformanceCounter(&qb);

                    double us = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                    latencies[q] = us;

                    std::vector<hnswlib::labeltype> res_ids;
                    uint32_t query_leaks = 0;
                    while (!result.empty()) {
                        hnswlib::labeltype id = result.top().second;
                        result.pop();
                        res_ids.push_back(id);
                        bool valid = (sel == 1U) ? (categories[id] == target_cat) : (categories[id] < target_cat);
                        if (!valid) {
                            query_leaks++;
                            total_leaks++;
                        }
                    }
                    std::reverse(res_ids.begin(), res_ids.end());

                    double r10 = recall_at_k(res_ids, ground_truth[sc][seli][q], 10);
                    double r100 = recall_at_k(res_ids, ground_truth[sc][seli][q], 100);
                    sum_r10 += r10;
                    sum_r100 += r100;

                    csv_per_query << header.seed_index << "," << scenario_name << "," << sel << ",hnswlib,"
                                  << ef << "," << q << "," << us << "," << r10 << "," << r100 << ","
                                  << res_ids.size() << "," << query_leaks << "\n";
                }
                QueryPerformanceCounter(&t1);

                double total_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
                double qps = total_sec > 0 ? (double)nq / total_sec : 0.0;

                std::sort(latencies.begin(), latencies.end());
                double p50 = latencies[(size_t)(nq * 0.50)];
                double p95 = latencies[(size_t)(nq * 0.95)];
                double p99 = latencies[(size_t)(nq * 0.99)];
                double avg_r10 = sum_r10 / (double)nq;
                double avg_r100 = sum_r100 / (double)nq;

                printf("hnswlib [ef=%3u] R10: %6.4f | R100: %6.4f | p50: %6.1f us | p95: %6.1f us | QPS: %5.1f | Leaks: %u\n",
                       ef, avg_r10, avg_r100, p50, p95, qps, total_leaks);

                csv_summary << header.seed_index << "," << scenario_name << "," << sel << ",hnswlib,"
                            << ef << "," << avg_r10 << "," << avg_r100 << ","
                            << p50 << "," << p95 << "," << p99 << "," << qps << ","
                            << build_sec << "," << index_mem_mb << "," << total_leaks << "\n";
                csv_summary.flush();
            }
        }
    }

    delete appr_alg;
    std::cout << "\n========================================================================\n";
    std::cout << "Evaluacion hnswlib completada exitosamente.\n";
    std::cout << "Resultados guardados en hito6/results/hnswlib_summary.csv y hnswlib_per_query.csv\n";
    return 0;
}
