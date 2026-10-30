#include "../common/dataset_format.h"
#include <windows.h>
#include <psapi.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <queue>

struct DistPair {
    float dist;
    uint32_t id;
    bool operator<(const DistPair &other) const {
        return dist < other.dist;
    }
};

static float l2_distance_avx2(const float *a, const float *b, uint32_t dim) {
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

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    std::string data_file = "hito6/data/dataset_seed1.bin";
    if (argc > 1) data_file = argv[1];

    std::cout << "========================================================================\n";
    std::cout << "Hito 6: Modalidad D — Exact Filtered Scan (Fuerza Bruta sobre Elegibles)\n";
    std::cout << "Archivo: " << data_file << "\n";
    std::cout << "========================================================================\n";

    FILE *f = fopen(data_file.c_str(), "rb");
    if (!f) return 1;

    DatasetHeader header;
    fread(&header, sizeof(header), 1, f);

    uint32_t nv = header.num_vectors;
    uint32_t dim = header.dim;
    uint32_t nq = header.num_queries;

    std::vector<float> vectors((size_t)nv * dim);
    std::vector<uint32_t> categories(nv);
    std::vector<float> queries_R((size_t)nq * dim);
    std::vector<float> queries_D((size_t)nq * dim);

    fread(vectors.data(), sizeof(float), (size_t)nv * dim, f);
    fread(categories.data(), sizeof(uint32_t), nv, f);
    fread(queries_R.data(), sizeof(float), (size_t)nq * dim, f);
    fread(queries_D.data(), sizeof(float), (size_t)nq * dim, f);
    fclose(f);

    CreateDirectoryA("hito6/results", NULL);
    std::ofstream csv("hito6/results/exact_scan_summary.csv");
    csv << "Seed,Scenario,Selectivity,Variant,p50_us,p95_us,p99_us,QPS,EligibleCount,DistanceComps\n";

    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);

    for (int sc = 0; sc < 2; sc++) {
        const char *scenario_name = (sc == 0) ? "R_Local" : "D_Adversarial";
        const float *q_base = (sc == 0) ? queries_R.data() : queries_D.data();
        std::cout << "\n==================== Escenario " << scenario_name << " ====================\n";

        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            uint32_t sel = header.selectivities[seli];
            uint32_t target_cat = header.target_categories[seli];

            /* Pre-indexar IDs elegibles para simular escaneo exacto sobre lista invertida/filtrada */
            std::vector<uint32_t> eligible_ids;
            eligible_ids.reserve((size_t)nv * sel / 100 + 100);
            for (uint32_t id = 0; id < nv; id++) {
                bool match = (sel == 1U) ? (categories[id] == target_cat) : (categories[id] < target_cat);
                if (match) eligible_ids.push_back(id);
            }

            std::vector<double> latencies(nq);
            QueryPerformanceCounter(&t0);

            for (uint32_t q = 0; q < nq; q++) {
                const float *query = q_base + (size_t)q * dim;
                LARGE_INTEGER qa, qb;
                QueryPerformanceCounter(&qa);

                std::priority_queue<DistPair> max_heap;
                for (uint32_t id : eligible_ids) {
                    float d = l2_distance_avx2(query, &vectors[(size_t)id * dim], dim);
                    if (max_heap.size() < 100) {
                        max_heap.push({d, id});
                    } else if (d < max_heap.top().dist) {
                        max_heap.pop();
                        max_heap.push({d, id});
                    }
                }
                QueryPerformanceCounter(&qb);
                latencies[q] = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
            }
            QueryPerformanceCounter(&t1);

            double total_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
            double qps = total_sec > 0 ? (double)nq / total_sec : 0.0;
            std::sort(latencies.begin(), latencies.end());
            double p50 = latencies[(size_t)(nq * 0.50)];
            double p95 = latencies[(size_t)(nq * 0.95)];
            double p99 = latencies[(size_t)(nq * 0.99)];

            printf("ExactScan [sel=%2u%% | elegibles=%5zu] p50: %6.1f us | p95: %6.1f us | QPS: %6.1f | dists: %zu\n",
                   sel, eligible_ids.size(), p50, p95, qps, eligible_ids.size());

            csv << header.seed_index << "," << scenario_name << "," << sel << ",ExactFilteredScan,"
                << p50 << "," << p95 << "," << p99 << "," << qps << ","
                << eligible_ids.size() << "," << eligible_ids.size() << "\n";
            csv.flush();
        }
    }

    std::cout << "\nModalidad D (ExactFilteredScan) completada exitosamente.\n";
    return 0;
}
