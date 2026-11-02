#include "../include/geometric_probe.h"
#include "../include/online_calibrator.h"
#include "../include/regret_metrics.h"
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
    std::cout << "Hito 9: Runner de Evaluacion OOD con Lazo Cerrado y Auditoria Online\n";
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
    csv_summary << "Dataset,Seed,K_Basins,VectorSigma,MixMode,Selectivity,Strategy,p50_us,p95_us,p99_us,MeanLatency_us,Recall10,Recall100,PctExact,PctHnsw,FallbackPct,PolicyRegretMean,PolicyRegret_p95,DiagnosticRegretMean,DiagnosticRegret_p95\n";

    std::ofstream csv_per_query(per_query_csv_path);
    csv_per_query << "Dataset,Seed,Selectivity,Strategy,QueryID,TrueLabel,Component,DetectedRegime,Latency_us,Recall10,Recall100,ChosenEngine,FallbackTriggered,FallbackReason,PriorState,PostState,EmpiricalLCB10,EmpiricalLCB100,PolicySignedRegret_us,PolicyPositiveRegret_us,DiagnosticPositiveRegret_us\n";

    struct StratDef {
        std::string name;
        enum { STRAT_EXACT, STRAT_HNSW, STRAT_STATIC15, STRAT_DISP } kind;
        DispatcherPolicy policy;
        double target_recall10;
        double target_recall100;
    };

    std::vector<StratDef> strats = {
        {"Fixed_ExactScan",       StratDef::STRAT_EXACT,    POLICY_EXACT,       1.0,  1.0},
        {"Fixed_hnswlib_ef50",    StratDef::STRAT_HNSW,     POLICY_APPROXIMATE, 0.0,  0.0},
        {"Static_Threshold_15pct",StratDef::STRAT_STATIC15, POLICY_APPROXIMATE, 0.0,  0.0},
        {"Dispatcher_EXACT",      StratDef::STRAT_DISP,     POLICY_EXACT,       1.0,  1.0},
        {"Dispatcher_BALANCED",   StratDef::STRAT_DISP,     POLICY_BALANCED,    0.95, 0.90},
        {"Dispatcher_APPROXIMATE",StratDef::STRAT_DISP,     POLICY_APPROXIMATE, 0.0,  0.0}
    };

    std::cout << "\n--- Ejecutando evaluacion en lazo cerrado sobre las 7 selectividades ---\n";

    for (uint32_t seli = 0; seli < num_sels; seli++) {
        uint32_t sel_deci = header.selectivities_deci_percent[seli];
        double sel_pct = (double)sel_deci / 10.0;
        uint32_t sel_bp = sel_deci * 10U; /* ej: 1500 para 15.0% */
        uint32_t target_cat = sel_deci * 10U;
        const auto &el_ids = eligible_per_sel[seli];
        uint32_t n_el = (uint32_t)el_ids.size();

        FilterQueryContext filter_ctx{target_cat, el_ids.data(), n_el};

        /* Warm-up y Pre-medicion robusta de latencias del oraculo:
           3 rondas con intercalado aleatorio para neutralizar sesgo de orden y calor de cache */
        std::vector<double> exact_raw_lat(nq), hnsw_raw_lat(nq);
        HybridNeighbor nbuf_ref[100];
        HybridSearchOutput out_ref{nbuf_ref, 100, 0};

        /* Warmup inicial */
        for (uint32_t w = 0; w < std::min(10U, nq); w++) {
            exact_scan_binding_search(&exact_ctx, queries[w].values, &filter_ctx, 100, 0, &out_ref);
            hnswlib_binding_search(&hnsw_ctx, queries[w].values, &filter_ctx, 100, 50, &out_ref);
        }

        const uint32_t NUM_REF_ROUNDS = 3;
        for (uint32_t q = 0; q < nq; q++) {
            const float *query = queries[q].values;
            double exact_samples[NUM_REF_ROUNDS];
            double hnsw_samples[NUM_REF_ROUNDS];

            for (uint32_t rnd = 0; rnd < NUM_REF_ROUNDS; rnd++) {
                LARGE_INTEGER qa, qb;
                /* Intercalado determinista pseudo-aleatorio para evitar sesgo de primer ejecutante */
                bool exact_first = (((q * 31U + rnd * 17U + seli) % 2U) == 0);

                if (exact_first) {
                    QueryPerformanceCounter(&qa);
                    exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out_ref);
                    QueryPerformanceCounter(&qb);
                    exact_samples[rnd] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;

                    QueryPerformanceCounter(&qa);
                    hnswlib_binding_search(&hnsw_ctx, query, &filter_ctx, 100, 50, &out_ref);
                    QueryPerformanceCounter(&qb);
                    hnsw_samples[rnd] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                } else {
                    QueryPerformanceCounter(&qa);
                    hnswlib_binding_search(&hnsw_ctx, query, &filter_ctx, 100, 50, &out_ref);
                    QueryPerformanceCounter(&qb);
                    hnsw_samples[rnd] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;

                    QueryPerformanceCounter(&qa);
                    exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out_ref);
                    QueryPerformanceCounter(&qb);
                    exact_samples[rnd] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                }
            }

            /* Estadistica robusta: mediana de las repeticiones intercaladas */
            std::sort(exact_samples, exact_samples + NUM_REF_ROUNDS);
            std::sort(hnsw_samples, hnsw_samples + NUM_REF_ROUNDS);
            exact_raw_lat[q] = exact_samples[NUM_REF_ROUNDS / 2];
            hnsw_raw_lat[q]  = hnsw_samples[NUM_REF_ROUNDS / 2];
        }

        for (const auto &st : strats) {
            std::vector<double> latencies(nq);
            std::vector<double> recalls10(nq), recalls100(nq);
            std::vector<double> policy_positive_regrets(nq), diagnostic_positive_regrets(nq);
            std::vector<double> policy_signed_regrets(nq);
            uint32_t exact_count = 0, hnsw_count = 0, fallback_count = 0;

            /* Instanciar calibradores online de lazo cerrado para cada estrategia */
            LatencyCalibratorConfig lcfg;
            latency_calibrator_default_config(&lcfg);
            lcfg.conservative_unseen = true;
            LatencyCalibrator *lat_cal = latency_calibrator_create(&lcfg);

            QualityAuditConfig qcfg;
            quality_audit_default_config(&qcfg);
            qcfg.target_recall10 = st.target_recall10;
            qcfg.target_recall100 = st.target_recall100;
            QualityAuditHarness *qual_audit = quality_audit_create(&qcfg);

            HybridNeighbor nbuf[100];
            HybridSearchOutput out{nbuf, 100, 0};
            HybridNeighbor shadow_nbuf[100];
            HybridSearchOutput shadow_out{shadow_nbuf, 100, 0};

            for (uint32_t q = 0; q < nq; q++) {
                const float *query = queries[q].values;
                QueryRegime reg = detected_regimes[q];
                HybridQueryResult meta{};
                LARGE_INTEGER qa, qb;
                double q_lat = 0.0;
                HybridEngineId chosen_id = ENGINE_EXACT_SCAN;
                bool fb_trig = false;
                std::string fb_reason = "NONE";

                /* Obtener estado previo del estrato */
                StratumCalibrator sinfo_prior{};
                latency_calibrator_get_stratum(lat_cal, reg, sel_bp, ENGINE_HNSWLIB, &sinfo_prior);
                StratumState prior_state = sinfo_prior.state;

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
                } else { /* STRAT_DISP con LAZO CERRADO */
                    if (st.policy == POLICY_EXACT) {
                        QueryPerformanceCounter(&qa);
                        exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out);
                        QueryPerformanceCounter(&qb);
                        q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                        chosen_id = ENGINE_EXACT_SCAN;
                    } else {
                        /* Evaluacion online de seguridad y fallback */
                        bool lat_fb = latency_calibrator_should_fallback(lat_cal, reg, sel_bp, ENGINE_HNSWLIB);
                        double empirical_lcb10 = 0.0, empirical_lcb100 = 0.0;
                        bool qual_fb = false;
                        if (st.policy == POLICY_BALANCED) {
                            qual_fb = quality_audit_should_fallback(qual_audit, reg, sel_bp, &empirical_lcb10, &empirical_lcb100);
                        }

                        bool force_fallback = (reg == QUERY_UNKNOWN || lat_fb || qual_fb);

                        if (force_fallback) {
                            fb_trig = true;
                            if (reg == QUERY_UNKNOWN) fb_reason = "OOD_UNKNOWN";
                            else if (qual_fb) fb_reason = "QUALITY_UNCALIBRATED_OR_DEGRADED";
                            else fb_reason = "LATENCY_DRIFT";

                            /* Servir la consulta mediante ExactScan seguro */
                            QueryPerformanceCounter(&qa);
                            exact_scan_binding_search(&exact_ctx, query, &filter_ctx, 100, 0, &out);
                            QueryPerformanceCounter(&qb);
                            q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                            chosen_id = ENGINE_EXACT_SCAN;

                            latency_calibrator_notify_fallback_served(lat_cal, reg, sel_bp, ENGINE_HNSWLIB);

                            /* Evaluacion Shadow de HNSW para recuperacion y acumulacion de evidencia */
                            if (prior_state == STRATUM_STATE_RECOVERY_PROBE || quality_audit_should_sample(qual_audit, q)) {
                                LARGE_INTEGER sa, sb;
                                QueryPerformanceCounter(&sa);
                                hnswlib_binding_search(&hnsw_ctx, query, &filter_ctx, 100, 50, &shadow_out);
                                QueryPerformanceCounter(&sb);
                                double s_lat = (double)(sb.QuadPart - sa.QuadPart) * 1e6 / (double)fq.QuadPart;
                                double s_r10 = compute_recall_at_k(shadow_out, ground_truth[seli][q], 10);
                                double s_r100 = compute_recall_at_k(shadow_out, ground_truth[seli][q], 100);

                                EngineEstimate h_est = cost_model_hnsw_predict(&hm, reg, sel_pct, 100);
                                DriftReason d_reason = DRIFT_NONE;
                                latency_calibrator_record(lat_cal, reg, sel_bp, ENGINE_HNSWLIB, s_lat, h_est.cost_us, &d_reason);
                                quality_audit_record(qual_audit, reg, sel_bp, s_r10, s_r100, s_lat);
                            }
                        } else {
                            /* Despacho nominal adaptativo */
                            QueryFeatures qf{n_el, nv, dim, reg, sel_pct};
                            QualityRequirements qr{st.policy, st.target_recall10, 0.0, 100};

                            QueryPerformanceCounter(&qa);
                            hybrid_search(&bindings, &qf, &qr, &em, &hm, query, &filter_ctx, &out, &meta);
                            QueryPerformanceCounter(&qb);
                            q_lat = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                            chosen_id = meta.chosen_engine;
                            fb_trig = meta.fallback_triggered;
                            if (fb_trig) fb_reason = "COST_MODEL_EXACT_PREFERRED";

                            /* Retroalimentacion online de latencia y calidad si se eligio HNSW */
                            if (chosen_id == ENGINE_HNSWLIB) {
                                EngineEstimate h_est = cost_model_hnsw_predict(&hm, reg, sel_pct, 100);
                                DriftReason d_reason = DRIFT_NONE;
                                latency_calibrator_record(lat_cal, reg, sel_bp, ENGINE_HNSWLIB, q_lat, h_est.cost_us, &d_reason);
                                if (quality_audit_should_sample(qual_audit, q)) {
                                    double r10_obs = compute_recall_at_k(out, ground_truth[seli][q], 10);
                                    double r100_obs = compute_recall_at_k(out, ground_truth[seli][q], 100);
                                    quality_audit_record(qual_audit, reg, sel_bp, r10_obs, r100_obs, 0.0);
                                }
                            }
                        }
                    }
                }

                latencies[q] = q_lat;
                if (chosen_id == ENGINE_EXACT_SCAN) exact_count++;
                else if (chosen_id == ENGINE_HNSWLIB) hnsw_count++;
                if (fb_trig) fallback_count++;

                double r10 = compute_recall_at_k(out, ground_truth[seli][q], 10);
                double r100 = compute_recall_at_k(out, ground_truth[seli][q], 100);
                recalls10[q] = r10;
                recalls100[q] = r100;

                /* Obtener estado posterior del estrato y LCB empíricos duales */
                StratumCalibrator sinfo_post{};
                latency_calibrator_get_stratum(lat_cal, reg, sel_bp, ENGINE_HNSWLIB, &sinfo_post);
                StratumState post_state = sinfo_post.state;

                double cur_lcb10 = 0.0, cur_lcb100 = 0.0;
                quality_audit_get_state(qual_audit, reg, sel_bp, &cur_lcb10, &cur_lcb100);

                /* Evaluacion estricta de Regret por Politica vs Diagnostico */
                EngineEstimate h_est_cand = cost_model_hnsw_predict(&hm, reg, sel_pct, 100);
                bool hnsw_admissible = false;
                if (st.policy == POLICY_APPROXIMATE) {
                    hnsw_admissible = true;
                } else if (st.policy == POLICY_BALANCED) {
                    hnsw_admissible = (cur_lcb10 >= st.target_recall10 && cur_lcb100 >= st.target_recall100) ||
                                      (h_est_cand.supported && h_est_cand.recall_lcb >= st.target_recall10 && h_est_cand.recall100 >= st.target_recall100);
                }

                RegretEvaluation r_eval = evaluate_regret(st.policy, q_lat, exact_raw_lat[q], hnsw_raw_lat[q],
                                                          hnsw_admissible, chosen_id);

                policy_signed_regrets[q] = r_eval.policy_signed_regret_us;
                policy_positive_regrets[q] = r_eval.policy_positive_regret_us;
                diagnostic_positive_regrets[q] = r_eval.diagnostic_positive_regret_us;

                csv_per_query << base_name << "," << header.seed << "," << sel_pct << ","
                              << st.name << "," << q << ","
                              << (queries[q].label == OOD_QUERY_LOCAL ? "LOCAL" : "ADVERSARIAL") << ","
                              << (queries[q].component == OOD_COMPONENT_LOCAL_BASIN ? "BASIN" : queries[q].component == OOD_COMPONENT_LOCAL_UNIFORM ? "LOCAL_UNIFORM" : "ADV_UNIFORM") << ","
                              << (reg == QUERY_LOCAL ? "LOCAL" : reg == QUERY_ADVERSARIAL ? "ADVERSARIAL" : "UNKNOWN") << ","
                              << q_lat << "," << r10 << "," << r100 << ","
                              << (chosen_id == ENGINE_EXACT_SCAN ? "EXACT" : "HNSW") << ","
                              << (fb_trig ? 1 : 0) << "," << fb_reason << ","
                              << prior_state << "," << post_state << "," << cur_lcb10 << "," << cur_lcb100 << ","
                              << r_eval.policy_signed_regret_us << "," << r_eval.policy_positive_regret_us << ","
                              << r_eval.diagnostic_positive_regret_us << "\n";
            }

            latency_calibrator_free(lat_cal);
            quality_audit_free(qual_audit);

            std::sort(latencies.begin(), latencies.end());
            std::sort(policy_positive_regrets.begin(), policy_positive_regrets.end());
            std::sort(diagnostic_positive_regrets.begin(), diagnostic_positive_regrets.end());

            double p50 = latencies[(size_t)(nq * 0.50)];
            double p95 = latencies[(size_t)(nq * 0.95)];
            double p99 = latencies[(size_t)(nq * 0.99)];
            double mean_lat = 0.0, mean_r10 = 0.0, mean_r100 = 0.0;
            double mean_pol_reg = 0.0, mean_diag_reg = 0.0;
            for (uint32_t i = 0; i < nq; i++) {
                mean_lat += latencies[i];
                mean_r10 += recalls10[i];
                mean_r100 += recalls100[i];
                mean_pol_reg += policy_positive_regrets[i];
                mean_diag_reg += diagnostic_positive_regrets[i];
            }
            mean_lat /= nq; mean_r10 /= nq; mean_r100 /= nq;
            mean_pol_reg /= nq; mean_diag_reg /= nq;

            double pol_reg_p95 = policy_positive_regrets[(size_t)(nq * 0.95)];
            double diag_reg_p95 = diagnostic_positive_regrets[(size_t)(nq * 0.95)];

            double pct_exact = (double)exact_count * 100.0 / nq;
            double pct_hnsw  = (double)hnsw_count * 100.0 / nq;
            double pct_fb    = (double)fallback_count * 100.0 / nq;

            csv_summary << base_name << "," << header.seed << "," << header.num_basins << ","
                        << header.vector_sigma << ","
                        << (header.mix_mode == OOD_MIX_R ? "R" : header.mix_mode == OOD_MIX_D ? "D" : "M50/50") << ","
                        << sel_pct << "," << st.name << "," << p50 << "," << p95 << "," << p99 << ","
                        << mean_lat << "," << mean_r10 * 100.0 << "," << mean_r100 * 100.0 << ","
                        << pct_exact << "," << pct_hnsw << "," << pct_fb << ","
                        << mean_pol_reg << "," << pol_reg_p95 << "," << mean_diag_reg << "," << diag_reg_p95 << "\n";

            printf("  [%-22s] Sel=%4.1f%% | p50=%6.1f us | R@10=%5.2f%% | Route(Ex/Hn)=%5.1f%%/%-5.1f%% | PolRegret_p95=%5.1f us\n",
                   st.name.c_str(), sel_pct, p50, mean_r10 * 100.0, pct_exact, pct_hnsw, pol_reg_p95);
        }
    }

    delete appr_alg;
    std::cout << "\nEvaluacion OOD completada con exito.\nResultados guardados en:\n  - "
              << summary_csv_path << "\n  - " << per_query_csv_path << "\n";
    return 0;
}
