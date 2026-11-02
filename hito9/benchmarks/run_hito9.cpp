#include "../include/geometric_probe.h"
#include "../datasets/ood_dataset_format.h"
#include "../../hito8/include/hybrid_engine.h"
#include "../../hito8/include/cost_model.h"
#include "../../hito8/include/engine_bindings.h"
#include "../../hito7/third_party/hnswlib/hnswlib/hnswlib.h"

#include <windows.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>

struct GtPair {
    float dist;
    uint32_t id;
};

static int cmp_gt_pair_asc(const void *a, const void *b) {
    float da = ((const GtPair *)a)->dist;
    float db = ((const GtPair *)b)->dist;
    return (da > db) - (da < db);
}

static inline float l2_dist_sq_avx2(const float *a, const float *b, uint32_t dim) {
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

struct QueryGT {
    uint32_t count;
    uint32_t ids[100];
    float dists[100];
};

static double compute_recall_at_k(const HybridSearchOutput &out, const QueryGT &gt, uint32_t k) {
    uint32_t n_gt = (gt.count < k) ? gt.count : k;
    if (n_gt == 0) return 1.0;
    uint32_t n_pred = (out.count < k) ? out.count : k;
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

    std::string data_file = "hito9/datasets/manifests/ood_seed8_k64_s009_m5050.bin";
    if (argc > 1) data_file = argv[1];

    std::cout << "========================================================================\n";
    std::cout << "Hito 9: Runner de Evaluacion OOD (Formato AOD9)\n";
    std::cout << "Archivo: " << data_file << "\n";
    std::cout << "========================================================================\n";

    FILE *f = fopen(data_file.c_str(), "rb");
    if (!f) {
        std::cerr << "Error abriendo archivo: " << data_file << "\n";
        return 1;
    }

    Hito9OODHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1 || header.magic != HITO9_OOD_MAGIC || header.version != HITO9_OOD_VERSION) {
        std::cerr << "Formato invalido en header AOD9.\n";
        fclose(f);
        return 2;
    }

    uint32_t nv = header.num_vectors;
    uint32_t dim = header.dimension;
    uint32_t nq = header.num_queries;
    uint32_t num_sels = header.num_selectivities;

    std::cout << "Cargando datos: " << nv << " vectores (dim=" << dim << "), "
              << nq << " consultas (K=" << header.num_basins << ", sigma=" << header.vector_sigma
              << ", mix=" << (header.mix_mode == OOD_MIX_R ? "R" : header.mix_mode == OOD_MIX_D ? "D" : "M50/50") << ")...\n";

    std::vector<float> vectors((size_t)nv * dim);
    std::vector<uint32_t> categories(nv);
    std::vector<Hito9OODQueryRecord> queries(nq);

    fread(vectors.data(), sizeof(float), (size_t)nv * dim, f);
    fread(categories.data(), sizeof(uint32_t), nv, f);
    fread(queries.data(), sizeof(Hito9OODQueryRecord), nq, f);
    fclose(f);

    /* 1. Construir Sonda Geometrica en Build-Time */
    std::cout << "Construyendo sonda geometrica (K=64 prototipos sobre vectores de construccion)...\n";
    GeometricProbe probe{};
    geometric_probe_build(&probe, vectors.data(), nv, dim, 64, 10, header.seed);

    /* 2. Evaluar Matriz de Confusion de la sonda sobre las consultas OOD */
    uint32_t tp = 0, fn = 0, fp = 0, tn = 0, unk_r = 0, unk_d = 0;
    uint32_t total_r = 0, total_d = 0;
    std::vector<QueryRegime> detected_regimes(nq);

    for (uint32_t q = 0; q < nq; q++) {
        float min_d = 0.0f;
        QueryRegime reg = geometric_probe_classify(&probe, queries[q].values, &min_d);
        detected_regimes[q] = reg;

        if (queries[q].label == OOD_QUERY_LOCAL) {
            total_r++;
            if (reg == QUERY_LOCAL) tp++;
            else if (reg == QUERY_ADVERSARIAL) fn++;
            else unk_r++;
        } else {
            total_d++;
            if (reg == QUERY_LOCAL) fp++;
            else if (reg == QUERY_ADVERSARIAL) tn++;
            else unk_d++;
        }
    }

    std::cout << "\n--- Matriz de Confusion de la Sonda Geometrica sobre este Dataset OOD ---\n";
    std::cout << "Consultas R reales (" << total_r << "): TP=" << tp << " (" << (double)tp*100.0/std::max(1U, total_r) << "%) | "
              << "FN=" << fn << " (" << (double)fn*100.0/std::max(1U, total_r) << "%) | "
              << "Unk=" << unk_r << " (" << (double)unk_r*100.0/std::max(1U, total_r) << "%)\n";
    std::cout << "Consultas D reales (" << total_d << "): TN=" << tn << " (" << (double)tn*100.0/std::max(1U, total_d) << "%) | "
              << "FP=" << fp << " (" << (double)fp*100.0/std::max(1U, total_d) << "%) | "
              << "Unk=" << unk_d << " (" << (double)unk_d*100.0/std::max(1U, total_d) << "%)\n";

    /* 3. Calcular Ground Truth exacto para cada selectividad */
    std::cout << "\nCalculando Ground Truth exacto para las " << num_sels << " selectividades...\n";
    std::vector<std::vector<uint32_t>> eligible_per_sel(num_sels);
    std::vector<std::vector<QueryGT>> ground_truth(num_sels, std::vector<QueryGT>(nq));

    GtPair *candidates = (GtPair *)malloc((size_t)nv * sizeof(GtPair));
    for (uint32_t seli = 0; seli < num_sels; seli++) {
        uint32_t sel_deci = header.selectivities_deci_percent[seli];
        uint32_t target_cat = sel_deci * 10U;

        eligible_per_sel[seli].reserve((size_t)nv * sel_deci / 1000 + 100);
        for (uint32_t i = 0; i < nv; i++) {
            if (categories[i] < target_cat) {
                eligible_per_sel[seli].push_back(i);
            }
        }
        uint32_t n_el = (uint32_t)eligible_per_sel[seli].size();

        for (uint32_t q = 0; q < nq; q++) {
            const float *query = queries[q].values;
            for (uint32_t idx = 0; idx < n_el; idx++) {
                uint32_t id = eligible_per_sel[seli][idx];
                candidates[idx].id = id;
                candidates[idx].dist = l2_dist_sq_avx2(vectors.data() + (size_t)id * dim, query, dim);
            }

            qsort(candidates, n_el, sizeof(GtPair), cmp_gt_pair_asc);
            uint32_t ret = (n_el < 100U) ? n_el : 100U;
            ground_truth[seli][q].count = ret;
            for (uint32_t k = 0; k < ret; k++) {
                ground_truth[seli][q].ids[k] = candidates[k].id;
                ground_truth[seli][q].dists[k] = candidates[k].dist;
            }
        }
    }
    free(candidates);

    /* 4. Construir indice HNSW */
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

    /* Contextos de motor y modelos */
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

    /* Preparar archivos CSV de resultados */
    CreateDirectoryA("hito9", NULL);
    CreateDirectoryA("hito9/results", NULL);
    std::string base_name = data_file.substr(data_file.find_last_of("/\\") + 1);
    std::string summary_csv_path = "hito9/results/" + base_name + "_summary.csv";
    std::string per_query_csv_path = "hito9/results/" + base_name + "_per_query.csv";

    std::ofstream csv_summary(summary_csv_path);
    csv_summary << "Dataset,Seed,K_Basins,VectorSigma,MixMode,Selectivity,Strategy,p50_us,p95_us,p99_us,MeanLatency_us,Recall10,Recall100,PctExact,PctHnsw,FallbackPct,SignedRegretMean,SignedRegret_p95,PositiveRegretMean,PositiveRegret_p95\n";

    std::ofstream csv_per_query(per_query_csv_path);
    csv_per_query << "Dataset,Seed,Selectivity,Strategy,QueryID,TrueLabel,Component,DetectedRegime,Latency_us,Recall10,Recall100,ChosenEngine,FallbackTriggered,SignedRegret_us,PositiveRegret_us\n";

    struct StratDef {
        std::string name;
        enum { STRAT_EXACT, STRAT_HNSW, STRAT_STATIC15, STRAT_DISP } kind;
        DispatcherPolicy policy;
        double target_recall;
    };

    std::vector<StratDef> strats = {
        {"Fixed_ExactScan",       StratDef::STRAT_EXACT,    POLICY_EXACT,       1.0},
        {"Fixed_hnswlib_ef50",    StratDef::STRAT_HNSW,     POLICY_APPROXIMATE, 0.0},
        {"Static_Threshold_15pct",StratDef::STRAT_STATIC15, POLICY_APPROXIMATE, 0.0},
        {"Dispatcher_EXACT",      StratDef::STRAT_DISP,     POLICY_EXACT,       1.0},
        {"Dispatcher_BALANCED",   StratDef::STRAT_DISP,     POLICY_BALANCED,    0.95},
        {"Dispatcher_APPROXIMATE",StratDef::STRAT_DISP,     POLICY_APPROXIMATE, 0.0}
    };

    std::cout << "\n--- Ejecutando evaluacion sobre las 7 selectividades ---\n";

    for (uint32_t seli = 0; seli < num_sels; seli++) {
        uint32_t sel_deci = header.selectivities_deci_percent[seli];
        double sel_pct = (double)sel_deci / 10.0;
        uint32_t target_cat = sel_deci * 10U;
        const auto &el_ids = eligible_per_sel[seli];
        uint32_t n_el = (uint32_t)el_ids.size();

        FilterQueryContext filter_ctx{target_cat, el_ids.data(), n_el};

        /* Pre-medir latencias crudas de ambos motores para el oraculo admisible */
        std::vector<double> exact_raw_lat(nq), hnsw_raw_lat(nq);
        HybridNeighbor nbuf_ref[100];
        HybridSearchOutput out_ref{nbuf_ref, 100, 0};

        for (uint32_t q = 0; q < nq; q++) {
            const float *query = queries[q].values;
            LARGE_INTEGER qa, qb;

            QueryPerformanceCounter(&qa);
            exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out_ref);
            QueryPerformanceCounter(&qb);
            exact_raw_lat[q] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;

            QueryPerformanceCounter(&qa);
            hnswlib_binding_search(&hnsw_ctx, query, &filter_ctx, 100, 50, &out_ref);
            QueryPerformanceCounter(&qb);
            hnsw_raw_lat[q] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
        }

        for (const auto &st : strats) {
            std::vector<double> latencies(nq);
            std::vector<double> recalls10(nq), recalls100(nq);
            std::vector<double> signed_regrets(nq), positive_regrets(nq);
            uint32_t exact_count = 0, hnsw_count = 0, fallback_count = 0;

            HybridNeighbor nbuf[100];
            HybridSearchOutput out{nbuf, 100, 0};

            for (uint32_t q = 0; q < nq; q++) {
                const float *query = queries[q].values;
                QueryRegime reg = detected_regimes[q];
                HybridQueryResult meta{};
                LARGE_INTEGER qa, qb;
                double q_lat = 0.0;
                HybridEngineId chosen_id = ENGINE_EXACT_SCAN;
                bool fb_trig = false;

                if (st.kind == StratDef::STRAT_EXACT) {
                    QueryPerformanceCounter(&qa);
                    exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out);
                    QueryPerformanceCounter(&qb);
                    q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                    chosen_id = ENGINE_EXACT_SCAN;
                } else if (st.kind == StratDef::STRAT_HNSW) {
                    QueryPerformanceCounter(&qa);
                    hnswlib_binding_search(&hnsw_ctx, query, &filter_ctx, 100, 50, &out);
                    QueryPerformanceCounter(&qb);
                    q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                    chosen_id = ENGINE_HNSWLIB;
                } else if (st.kind == StratDef::STRAT_STATIC15) {
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
                } else { /* STRAT_DISP */
                    QueryFeatures qf{n_el, nv, dim, reg, sel_pct};
                    QualityRequirements qr{st.policy, st.target_recall, 0.0, 100};

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

                double r10 = compute_recall_at_k(out, ground_truth[seli][q], 10);
                double r100 = compute_recall_at_k(out, ground_truth[seli][q], 100);
                recalls10[q] = r10;
                recalls100[q] = r100;

                /* Definicion estricta ex-ante de Oraculo Admisible T*(q):
                 * EXACT: T* = T_exact
                 * APPROXIMATE: T* = min(T_exact, T_hnsw)
                 * BALANCED: Si el modelo calibrado ex-ante para este régimen y estrato
                 *           soporta HNSW con LCB >= 0.95, T* = min(T_exact, T_hnsw).
                 *           De lo contrario, HNSW no es admisible ex-ante, T* = T_exact.
                 */
                double t_star = exact_raw_lat[q];
                if (st.policy == POLICY_APPROXIMATE) {
                    t_star = std::min(exact_raw_lat[q], hnsw_raw_lat[q]);
                } else if (st.policy == POLICY_BALANCED) {
                    EngineEstimate h_est = cost_model_hnsw_predict(&hm, reg, sel_pct, 100);
                    if (h_est.supported && h_est.recall_lcb >= 0.95) {
                        t_star = std::min(exact_raw_lat[q], hnsw_raw_lat[q]);
                    } else {
                        t_star = exact_raw_lat[q];
                    }
                } else {
                    t_star = exact_raw_lat[q];
                }

                double s_reg = q_lat - t_star;
                double p_reg = (s_reg > 0.0) ? s_reg : 0.0;
                signed_regrets[q] = s_reg;
                positive_regrets[q] = p_reg;

                csv_per_query << base_name << "," << header.seed << "," << sel_pct << ","
                              << st.name << "," << q << ","
                              << (queries[q].label == OOD_QUERY_LOCAL ? "LOCAL" : "ADVERSARIAL") << ","
                              << (queries[q].component == OOD_COMPONENT_LOCAL_BASIN ? "BASIN" : queries[q].component == OOD_COMPONENT_LOCAL_UNIFORM ? "LOCAL_UNIFORM" : "ADV_UNIFORM") << ","
                              << (reg == QUERY_LOCAL ? "LOCAL" : reg == QUERY_ADVERSARIAL ? "ADVERSARIAL" : "UNKNOWN") << ","
                              << q_lat << "," << r10 << "," << r100 << ","
                              << (chosen_id == ENGINE_EXACT_SCAN ? "EXACT" : "HNSW") << ","
                              << (fb_trig ? 1 : 0) << "," << s_reg << "," << p_reg << "\n";
            }

            std::sort(latencies.begin(), latencies.end());
            std::sort(signed_regrets.begin(), signed_regrets.end());
            std::sort(positive_regrets.begin(), positive_regrets.end());

            double p50 = latencies[(size_t)(nq * 0.50)];
            double p95 = latencies[(size_t)(nq * 0.95)];
            double p99 = latencies[(size_t)(nq * 0.99)];
            double mean_lat = 0.0, mean_r10 = 0.0, mean_r100 = 0.0;
            double mean_s_reg = 0.0, mean_p_reg = 0.0;
            for (uint32_t i = 0; i < nq; i++) {
                mean_lat += latencies[i];
                mean_r10 += recalls10[i];
                mean_r100 += recalls100[i];
                mean_s_reg += signed_regrets[i];
                mean_p_reg += positive_regrets[i];
            }
            mean_lat /= nq; mean_r10 /= nq; mean_r100 /= nq;
            mean_s_reg /= nq; mean_p_reg /= nq;

            double s_reg_p95 = signed_regrets[(size_t)(nq * 0.95)];
            double p_reg_p95 = positive_regrets[(size_t)(nq * 0.95)];

            double pct_exact = (double)exact_count * 100.0 / nq;
            double pct_hnsw  = (double)hnsw_count * 100.0 / nq;
            double pct_fb    = (double)fallback_count * 100.0 / nq;

            csv_summary << base_name << "," << header.seed << "," << header.num_basins << ","
                        << header.vector_sigma << ","
                        << (header.mix_mode == OOD_MIX_R ? "R" : header.mix_mode == OOD_MIX_D ? "D" : "M50/50") << ","
                        << sel_pct << "," << st.name << "," << p50 << "," << p95 << "," << p99 << ","
                        << mean_lat << "," << mean_r10 * 100.0 << "," << mean_r100 * 100.0 << ","
                        << pct_exact << "," << pct_hnsw << "," << pct_fb << ","
                        << mean_s_reg << "," << s_reg_p95 << "," << mean_p_reg << "," << p_reg_p95 << "\n";

            printf("  [%-22s] Sel=%4.1f%% | p50=%6.1f us | R@10=%5.2f%% | Route(Ex/Hn)=%5.1f%%/%-5.1f%% | +Regret_p95=%5.1f us\n",
                   st.name.c_str(), sel_pct, p50, mean_r10 * 100.0, pct_exact, pct_hnsw, p_reg_p95);
        }
    }

    delete appr_alg;
    std::cout << "\nEvaluacion OOD completada con exito.\nResultados guardados en:\n  - "
              << summary_csv_path << "\n  - " << per_query_csv_path << "\n";
    return 0;
}
