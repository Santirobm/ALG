#include "../include/hybrid_engine.h"
#include "../include/cost_model.h"
#include "../include/engine_bindings.h"
#include "../../hito7/common/dataset_format.h"
#include "../../hito7/third_party/hnswlib/hnswlib/hnswlib.h"

#include <windows.h>
#include <psapi.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>

struct QueryGT {
    uint32_t count;
    uint32_t ids[TOP_K_GT];
    float dists[TOP_K_GT];
};

static double compute_recall_at_k(const HybridSearchOutput &out, const QueryGT &gt, uint32_t k) {
    uint32_t n_gt = gt.count < k ? gt.count : k;
    if (n_gt == 0) return 1.0;
    uint32_t n_pred = out.count < k ? out.count : k;
    uint32_t match = 0;
    for (uint32_t i = 0; i < n_pred; i++) {
        for (uint32_t j = 0; j < n_gt; j++) {
            if (out.neighbors[i].id == gt.ids[j]) {
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
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    std::string data_file = "hito8/data/dataset_seed6.bin";
    if (argc > 1) data_file = argv[1];

    std::cout << "========================================================================\n";
    std::cout << "Hito 8: Evaluacion Ciega del Selector Hibrido Adaptativo\n";
    std::cout << "Dataset: " << data_file << "\n";
    std::cout << "========================================================================\n";

    FILE *f = fopen(data_file.c_str(), "rb");
    if (!f) {
        std::cerr << "Error abriendo dataset: " << data_file << "\n";
        return 1;
    }

    DatasetHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1 || header.magic != DATASET_MAGIC || header.version != DATASET_VERSION) {
        std::cerr << "Encabezado invalido en " << data_file << "\n";
        fclose(f);
        return 2;
    }

    uint32_t nv = header.num_vectors;
    uint32_t dim = header.dim;
    uint32_t nq = header.num_queries;

    std::cout << "Cargando dataset: " << nv << " vectores, dim=" << dim << ", " << nq << " consultas/escenario...\n";
    std::vector<float> vectors((size_t)nv * dim);
    std::vector<uint32_t> categories(nv);
    std::vector<float> queries_R((size_t)nq * dim);
    std::vector<float> queries_D((size_t)nq * dim);

    fread(vectors.data(), sizeof(float), (size_t)nv * dim, f);
    fread(categories.data(), sizeof(uint32_t), nv, f);
    fread(queries_R.data(), sizeof(float), (size_t)nq * dim, f);
    fread(queries_D.data(), sizeof(float), (size_t)nq * dim, f);

    /* Cargar Ground Truth exacto para los 14 bloques */
    std::vector<std::vector<std::vector<QueryGT>>> ground_truth(
        2, std::vector<std::vector<QueryGT>>(NUM_SELECTIVITIES, std::vector<QueryGT>(nq)));

    for (int sc = 0; sc < 2; sc++) {
        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            for (uint32_t q = 0; q < nq; q++) {
                uint32_t ret = 0;
                fread(&ret, sizeof(uint32_t), 1, f);
                ground_truth[sc][seli][q].count = ret;
                fread(ground_truth[sc][seli][q].ids, sizeof(uint32_t), TOP_K_GT, f);
                fread(ground_truth[sc][seli][q].dists, sizeof(float), TOP_K_GT, f);
            }
        }
    }
    fclose(f);

    /* 1. Construir indice HNSW para hnswlib */
    std::cout << "Construyendo indice hnswlib (M=16, efConstruction=200)...\n";
    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&t0);

    hnswlib::L2Space space(dim);
    auto *appr_alg = new hnswlib::HierarchicalNSW<float>(&space, nv, 16, 200);
    for (size_t i = 0; i < nv; i++) {
        appr_alg->addPoint(vectors.data() + i * dim, (hnswlib::labeltype)i);
    }
    QueryPerformanceCounter(&t1);
    double build_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
    std::cout << "hnswlib construido en " << build_sec << " s\n";

    /* 2. Pre-indexar listas de IDs elegibles por selectividad */
    std::vector<std::vector<uint32_t>> eligible_per_sel(NUM_SELECTIVITIES);
    for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
        uint32_t target_cat = header.target_categories[seli];
        eligible_per_sel[seli].reserve((size_t)nv * header.selectivities[seli] / 1000 + 100);
        for (uint32_t i = 0; i < nv; i++) {
            if (categories[i] < target_cat) {
                eligible_per_sel[seli].push_back(i);
            }
        }
    }

    /* 3. Configurar contextos de motor y sonda de características */
    ExactEngineContext exact_ctx{vectors.data(), categories.data(), nv, dim};
    HnswlibEngineContext hnsw_ctx{appr_alg, categories.data(), nv, dim};

    HybridEngineBindings bindings{};
    bindings.exact_scan = {&exact_ctx, exact_scan_binding_search, 0, true};
    bindings.hnswlib    = {&hnsw_ctx, hnswlib_binding_search, 50, true};
    bindings.alg_h2_at  = {nullptr, nullptr, 512, false};

    ExactCostModel em{};
    cost_model_default_exact(&em);

    HnswCostModel hm{};
    cost_model_default_hnsw(&hm);

    /* Sonda de 32 centroides para detección de régimen */
    const uint32_t NUM_PROBES = 32U;
    std::vector<float> probe_centroids((size_t)NUM_PROBES * dim);
    for (uint32_t p = 0; p < NUM_PROBES; p++) {
        uint32_t sample_id = (uint32_t)((size_t)p * nv / NUM_PROBES);
        std::copy(vectors.begin() + (size_t)sample_id * dim,
                  vectors.begin() + (size_t)(sample_id + 1) * dim,
                  probe_centroids.begin() + (size_t)p * dim);
    }
    QueryFeatureProbe probe{probe_centroids.data(), NUM_PROBES, dim, 0.85f};

    /* Preparar archivos CSV de salida */
    CreateDirectoryA("hito8", NULL);
    CreateDirectoryA("hito8/results", NULL);
    std::string summary_csv_path = "hito8/results/hito8_seed" + std::to_string(header.seed_index) + "_summary.csv";
    std::string per_query_csv_path = "hito8/results/hito8_seed" + std::to_string(header.seed_index) + "_per_query.csv";

    std::ofstream csv_summary(summary_csv_path);
    csv_summary << "Seed,Scenario,Selectivity,Strategy,p50_us,p95_us,p99_us,MeanLatency_us,Recall10,Recall100,PctExact,PctHnsw,FallbackPct,MeanRegret_us,p95Regret_us\n";

    std::ofstream csv_per_query(per_query_csv_path);
    csv_per_query << "Seed,Scenario,Selectivity,Strategy,QueryID,Latency_us,Recall10,Recall100,ChosenEngine,FallbackTriggered,Regret_us\n";

    struct StrategyConfig {
        std::string name;
        enum { STRAT_EXACT_SCAN, STRAT_HNSWLIB, STRAT_STATIC_15PCT, STRAT_DISPATCHER } kind;
        DispatcherPolicy policy;
        double target_recall;
    };

    std::vector<StrategyConfig> strategies = {
        {"Fixed_ExactScan",       StrategyConfig::STRAT_EXACT_SCAN,   POLICY_EXACT,       1.0},
        {"Fixed_hnswlib_ef50",    StrategyConfig::STRAT_HNSWLIB,      POLICY_APPROXIMATE, 0.0},
        {"Static_Threshold_15pct",StrategyConfig::STRAT_STATIC_15PCT, POLICY_APPROXIMATE, 0.0},
        {"Dispatcher_EXACT",      StrategyConfig::STRAT_DISPATCHER,   POLICY_EXACT,       1.0},
        {"Dispatcher_BALANCED",   StrategyConfig::STRAT_DISPATCHER,   POLICY_BALANCED,    0.95},
        {"Dispatcher_APPROXIMATE",StrategyConfig::STRAT_DISPATCHER,   POLICY_APPROXIMATE, 0.0}
    };

    std::cout << "\n--- Iniciando evaluacion de estrategias sobre 14 estratos ---\n";

    for (int sc = 0; sc < 2; sc++) {
        const char *scenario_name = (sc == 0) ? "R_Local" : "D_Adversarial";
        const float *q_base = (sc == 0) ? queries_R.data() : queries_D.data();
        std::cout << "\n==================== Escenario: " << scenario_name << " ====================\n";

        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            uint32_t sel = header.selectivities[seli];
            double sel_pct = (double)sel / 10.0;
            uint32_t target_cat = header.target_categories[seli];
            const auto &el_ids = eligible_per_sel[seli];
            uint32_t n_el = (uint32_t)el_ids.size();

            FilterQueryContext filter_ctx{target_cat, el_ids.data(), n_el};

            /* Pre-computar latencias crudas y recall de ambos motores para referencia de oráculo admissible */
            std::vector<double> exact_raw_lat(nq), hnsw_raw_lat(nq);
            std::vector<double> hnsw_raw_r10(nq);

            HybridNeighbor nbuf_ref[100];
            HybridSearchOutput out_ref{nbuf_ref, 100, 0};

            for (uint32_t q = 0; q < nq; q++) {
                const float *query = q_base + (size_t)q * dim;
                LARGE_INTEGER qa, qb;

                /* ExactScan raw */
                QueryPerformanceCounter(&qa);
                exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out_ref);
                QueryPerformanceCounter(&qb);
                exact_raw_lat[q] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;

                /* HNSW raw */
                QueryPerformanceCounter(&qa);
                hnswlib_binding_search(&hnsw_ctx, query, &filter_ctx, 100, 50, &out_ref);
                QueryPerformanceCounter(&qb);
                hnsw_raw_lat[q] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                hnsw_raw_r10[q] = compute_recall_at_k(out_ref, ground_truth[sc][seli][q], 10);
            }

            for (const auto &strat : strategies) {
                std::vector<double> latencies(nq);
                std::vector<double> recalls10(nq), recalls100(nq);
                std::vector<double> regrets(nq);
                uint32_t exact_count = 0, hnsw_count = 0, fallback_count = 0;

                HybridNeighbor nbuf[100];
                HybridSearchOutput out{nbuf, 100, 0};

                for (uint32_t q = 0; q < nq; q++) {
                    const float *query = q_base + (size_t)q * dim;
                    HybridQueryResult meta{};
                    LARGE_INTEGER qa, qb;
                    double q_lat = 0.0;
                    HybridEngineId chosen_id = ENGINE_EXACT_SCAN;
                    bool fb_trig = false;

                    if (strat.kind == StrategyConfig::STRAT_EXACT_SCAN) {
                        QueryPerformanceCounter(&qa);
                        exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out);
                        QueryPerformanceCounter(&qb);
                        q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                        chosen_id = ENGINE_EXACT_SCAN;
                    } else if (strat.kind == StrategyConfig::STRAT_HNSWLIB) {
                        QueryPerformanceCounter(&qa);
                        hnswlib_binding_search(&hnsw_ctx, query, &filter_ctx, 100, 50, &out);
                        QueryPerformanceCounter(&qb);
                        q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                        chosen_id = ENGINE_HNSWLIB;
                    } else if (strat.kind == StrategyConfig::STRAT_STATIC_15PCT) {
                        if (sel_pct <= 15.0) {
                            QueryPerformanceCounter(&qa);
                            exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out);
                            QueryPerformanceCounter(&qb);
                            chosen_id = ENGINE_EXACT_SCAN;
                        } else {
                            QueryPerformanceCounter(&qa);
                            hnswlib_binding_search(&hnsw_ctx, query, &filter_ctx, 100, 50, &out);
                            QueryPerformanceCounter(&qb);
                            chosen_id = ENGINE_HNSWLIB;
                        }
                        q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                    } else { /* STRAT_DISPATCHER */
                        QueryRegime reg = extract_query_regime(&probe, query);
                        QueryFeatures qf{n_el, nv, dim, reg, sel_pct};
                        QualityRequirements qr{strat.policy, strat.target_recall, 0.0, 100};

                        QueryPerformanceCounter(&qa);
                        hybrid_search(&bindings, &qf, &qr, &em, &hm, query, &filter_ctx, &out, &meta);
                        QueryPerformanceCounter(&qb);
                        q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                        chosen_id = meta.chosen_engine;
                        fb_trig = meta.fallback_triggered;
                    }

                    latencies[q] = q_lat;
                    if (chosen_id == ENGINE_EXACT_SCAN) exact_count++;
                    else if (chosen_id == ENGINE_HNSWLIB) hnsw_count++;
                    if (fb_trig) fallback_count++;

                    double r10 = compute_recall_at_k(out, ground_truth[sc][seli][q], 10);
                    double r100 = compute_recall_at_k(out, ground_truth[sc][seli][q], 100);
                    recalls10[q] = r10;
                    recalls100[q] = r100;

                    /* Computar regret admisible: R_pi(q) = T_selector - min_{m in M_adm} T_m */
                    double t_star = exact_raw_lat[q];
                    if (strat.policy == POLICY_APPROXIMATE) {
                        t_star = std::min(exact_raw_lat[q], hnsw_raw_lat[q]);
                    } else if (strat.policy == POLICY_BALANCED) {
                        if (hnsw_raw_r10[q] >= 0.95) {
                            t_star = std::min(exact_raw_lat[q], hnsw_raw_lat[q]);
                        } else {
                            t_star = exact_raw_lat[q];
                        }
                    } else { /* POLICY_EXACT */
                        t_star = exact_raw_lat[q];
                    }
                    double reg_val = q_lat - t_star;
                    if (reg_val < 0.0) reg_val = 0.0;
                    regrets[q] = reg_val;

                    csv_per_query << header.seed_index << "," << scenario_name << "," << sel_pct << ","
                                  << strat.name << "," << q << "," << q_lat << "," << r10 << "," << r100 << ","
                                  << (chosen_id == ENGINE_EXACT_SCAN ? "EXACT" : "HNSW") << ","
                                  << (fb_trig ? 1 : 0) << "," << reg_val << "\n";
                }

                std::sort(latencies.begin(), latencies.end());
                std::sort(regrets.begin(), regrets.end());
                double p50 = latencies[(size_t)(nq * 0.50)];
                double p95 = latencies[(size_t)(nq * 0.95)];
                double p99 = latencies[(size_t)(nq * 0.99)];
                double mean_lat = 0.0, mean_r10 = 0.0, mean_r100 = 0.0, mean_regret = 0.0;
                for (uint32_t i = 0; i < nq; i++) {
                    mean_lat += latencies[i];
                    mean_r10 += recalls10[i];
                    mean_r100 += recalls100[i];
                    mean_regret += regrets[i];
                }
                mean_lat /= nq; mean_r10 /= nq; mean_r100 /= nq; mean_regret /= nq;
                double p95_regret = regrets[(size_t)(nq * 0.95)];

                double pct_exact = (double)exact_count * 100.0 / nq;
                double pct_hnsw  = (double)hnsw_count * 100.0 / nq;
                double pct_fb    = (double)fallback_count * 100.0 / nq;

                csv_summary << header.seed_index << "," << scenario_name << "," << sel_pct << ","
                            << strat.name << "," << p50 << "," << p95 << "," << p99 << "," << mean_lat << ","
                            << mean_r10 * 100.0 << "," << mean_r100 * 100.0 << "," << pct_exact << ","
                            << pct_hnsw << "," << pct_fb << "," << mean_regret << "," << p95_regret << "\n";

                printf("  [%-22s] Sel=%4.1f%% | p50=%6.1f us | R@10=%5.2f%% | Route(Ex/Hn)=%5.1f%%/%-5.1f%% | Regret_p95=%5.1f us\n",
                       strat.name.c_str(), sel_pct, p50, mean_r10 * 100.0, pct_exact, pct_hnsw, p95_regret);
            }
        }
    }

    delete appr_alg;
    std::cout << "\nEvaluacion Hito 8 completada con exito.\nResultados guardados en:\n  - "
              << summary_csv_path << "\n  - " << per_query_csv_path << "\n";
    return 0;
}
