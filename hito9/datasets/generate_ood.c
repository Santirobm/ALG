#include "ood_dataset_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#define DEFAULT_NV 100000U
#define DEFAULT_NQ 1000U
#define DEFAULT_K 64U
#define DEFAULT_SIGMA 0.09f
#define DEFAULT_LOCAL_QUERY_SIGMA 0.05f
#define DEFAULT_SEED 0x9A71D00D2026ULL
#define NUM_SELECTIVITIES 7U

static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float uniform01(uint64_t *s) {
    return (float)(rng_next(s) >> 40) / 16777216.0f;
}

static float gaussian01(uint64_t *s) {
    float u1 = uniform01(s);
    float u2 = uniform01(s);
    if (u1 < 1.0e-7f) u1 = 1.0e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864769f * u2);
}

static void normalize(float *v) {
    double n2 = 0.0;
    for (uint32_t d = 0; d < HITO9_OOD_DIM; ++d) n2 += (double)v[d] * v[d];
    float inv = (float)(1.0 / sqrt(n2 > 1.0e-20 ? n2 : 1.0));
    for (uint32_t d = 0; d < HITO9_OOD_DIM; ++d) v[d] *= inv;
}

static int parse_mix(const char *s, OODMixMode *mode) {
    if (!strcmp(s, "R") || !strcmp(s, "r")) *mode = OOD_MIX_R;
    else if (!strcmp(s, "D") || !strcmp(s, "d")) *mode = OOD_MIX_D;
    else if (!strcmp(s, "M50/50") || !strcmp(s, "m50/50")) *mode = OOD_MIX_50_50;
    else return 0;
    return 1;
}

static void usage(const char *exe) {
    fprintf(stderr,
        "Usage: %s [--seed N] [--num-vectors N] [--num-queries N]\n"
        "          [--basins 16|64|128] [--sigma 0.03|0.09|0.25]\n"
        "          [--mix R|D|M50/50] [--out FILE]\n", exe);
}

int main(int argc, char **argv) {
    uint64_t seed = DEFAULT_SEED;
    uint32_t nv = DEFAULT_NV, nq = DEFAULT_NQ, k = DEFAULT_K;
    float sigma = DEFAULT_SIGMA;
    OODMixMode mix = OOD_MIX_R;
    const char *out_path = "hito9/datasets/ood_dataset.bin";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--num-vectors") && i + 1 < argc) nv = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--num-queries") && i + 1 < argc) nq = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--basins") && i + 1 < argc) k = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--sigma") && i + 1 < argc) sigma = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--mix") && i + 1 < argc) {
            if (!parse_mix(argv[++i], &mix)) { usage(argv[0]); return 2; }
        }
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else { usage(argv[0]); return 2; }
    }

    if (nv < 128 || nq == 0 || (k != 16 && k != 64 && k != 128) ||
        !(sigma == 0.03f || sigma == 0.09f || sigma == 0.25f)) {
        fprintf(stderr, "Invalid parameters: require nv>=128, nq>0, K in {16,64,128}, sigma in {0.03,0.09,0.25}.\n");
        return 2;
    }

    float *centroids = (float *)malloc((size_t)k * HITO9_OOD_DIM * sizeof(float));
    float *vectors = (float *)malloc((size_t)nv * HITO9_OOD_DIM * sizeof(float));
    uint32_t *categories = (uint32_t *)malloc((size_t)nv * sizeof(uint32_t));
    uint32_t *basin_nodes = (uint32_t *)malloc((size_t)k * sizeof(uint32_t));
    Hito9OODQueryRecord *queries = (Hito9OODQueryRecord *)calloc(nq, sizeof(Hito9OODQueryRecord));
    if (!centroids || !vectors || !categories || !basin_nodes || !queries) {
        fprintf(stderr, "Allocation failed.\n");
        free(centroids); free(vectors); free(categories); free(basin_nodes); free(queries);
        return 1;
    }

    uint64_t centroid_rng = seed ^ 0x9988776655443322ULL;
    for (uint32_t c = 0; c < k; ++c) {
        float *center = centroids + (size_t)c * HITO9_OOD_DIM;
        for (uint32_t d = 0; d < HITO9_OOD_DIM; ++d) center[d] = 2.0f * uniform01(&centroid_rng) - 1.0f;
        normalize(center);
    }

    uint64_t point_rng = seed ^ 0x3344556677889900ULL;
    for (uint32_t c = 0; c < k; ++c) basin_nodes[c] = UINT32_MAX;
    for (uint32_t i = 0; i < nv; ++i) {
        uint32_t c = (uint32_t)(rng_next(&point_rng) % k);
        const float *center = centroids + (size_t)c * HITO9_OOD_DIM;
        if (basin_nodes[c] == UINT32_MAX) basin_nodes[c] = i;
        float *v = vectors + (size_t)i * HITO9_OOD_DIM;
        for (uint32_t d = 0; d < HITO9_OOD_DIM; ++d) v[d] = center[d] + sigma * gaussian01(&point_rng);
        normalize(v);

        /* Deterministic category hash, matching Hito 8's 0..9999 domain. */
        uint64_t z = seed + (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        categories[i] = (uint32_t)(z % 10000ULL);
    }

    for (uint32_t b = 0; b < k; ++b) {
        if (basin_nodes[b] == UINT32_MAX) {
            fprintf(stderr, "Generated corpus did not populate basin %u.\\n", b);
            free(centroids); free(vectors); free(categories); free(basin_nodes); free(queries);
            return 1;
        }
    }

    uint32_t local_basin_count = 0, local_uniform_count = 0, adversarial_count = 0;
    uint64_t query_rng = seed ^ 0xFEEDFACECAFEBEEFULL;
    const uint32_t local_count = (mix == OOD_MIX_R) ? nq :
                                (mix == OOD_MIX_D) ? 0U : (nq / 2U);
    for (uint32_t q = 0; q < nq; ++q) {
        int is_local = q < local_count;
        if (mix == OOD_MIX_R) is_local = 1;
        if (mix == OOD_MIX_D) is_local = 0;

        Hito9OODQueryRecord *rec = &queries[q];
        rec->label = is_local ? OOD_QUERY_LOCAL : OOD_QUERY_ADVERSARIAL;

        /* R distribution: 85% near one of K sampled basin nodes, 15% uniform. */
        int concentrated = is_local && ((rng_next(&query_rng) % 100U) < 85U);
        if (!is_local) { rec->component = OOD_COMPONENT_ADVERSARIAL_UNIFORM; adversarial_count++; }
        else if (concentrated) { rec->component = OOD_COMPONENT_LOCAL_BASIN; local_basin_count++; }
        else { rec->component = OOD_COMPONENT_LOCAL_UNIFORM; local_uniform_count++; }
        if (concentrated) {
            uint32_t b = (uint32_t)(rng_next(&query_rng) % k);
            const float *base = vectors + (size_t)basin_nodes[b] * HITO9_OOD_DIM;
            for (uint32_t d = 0; d < HITO9_OOD_DIM; ++d)
                rec->values[d] = base[d] + DEFAULT_LOCAL_QUERY_SIGMA * gaussian01(&query_rng);
        } else {
            for (uint32_t d = 0; d < HITO9_OOD_DIM; ++d)
                rec->values[d] = 2.0f * uniform01(&query_rng) - 1.0f;
        }
        normalize(rec->values);
    }

    /* Deterministic Fisher-Yates shuffle prevents block-order/cache bias in M50/50. */
    for (uint32_t i = nq - 1U; i > 0U; --i) {
        uint32_t j = (uint32_t)(rng_next(&query_rng) % (uint64_t)(i + 1U));
        Hito9OODQueryRecord tmp = queries[i];
        queries[i] = queries[j];
        queries[j] = tmp;
    }

    Hito9OODHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = HITO9_OOD_MAGIC;
    hdr.version = HITO9_OOD_VERSION;
    hdr.seed = seed;
    hdr.num_vectors = nv;
    hdr.dimension = HITO9_OOD_DIM;
    hdr.num_queries = nq;
    hdr.num_basins = k;
    hdr.vector_sigma = sigma;
    hdr.local_query_sigma = DEFAULT_LOCAL_QUERY_SIGMA;
    hdr.mix_mode = (uint32_t)mix;
    hdr.local_fraction_percent = (mix == OOD_MIX_R) ? 100U : (mix == OOD_MIX_D) ? 0U : 50U;
    hdr.local_basin_fraction_percent = 85U;
    hdr.num_selectivities = NUM_SELECTIVITIES;
    const uint32_t sels[NUM_SELECTIVITIES] = {100, 125, 150, 175, 200, 225, 250};
    memcpy(hdr.selectivities_deci_percent, sels, sizeof(sels));

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s: %s\n", out_path, strerror(errno));
        free(centroids); free(vectors); free(categories); free(basin_nodes); free(queries);
        return 1;
    }
    int ok = fwrite(&hdr, sizeof(hdr), 1, f) == 1 &&
             fwrite(vectors, sizeof(float), (size_t)nv * HITO9_OOD_DIM, f) == (size_t)nv * HITO9_OOD_DIM &&
             fwrite(categories, sizeof(uint32_t), nv, f) == nv &&
             fwrite(queries, sizeof(Hito9OODQueryRecord), nq, f) == nq;
    if (fclose(f) != 0) ok = 0;

    if (!ok) {
        fprintf(stderr, "Write failed for %s.\n", out_path);
        remove(out_path);
        free(centroids); free(vectors); free(categories); free(basin_nodes); free(queries);
        return 1;
    }

    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path), "%s.manifest.csv", out_path);
    FILE *mf = fopen(manifest_path, "w");
    if (!mf) {
        fprintf(stderr, "Dataset written, but manifest could not be opened: %s\n", manifest_path);
        free(centroids); free(vectors); free(categories); free(basin_nodes); free(queries);
        return 1;
    }
    fprintf(mf, "seed,num_vectors,num_queries,dimension,basins,vector_sigma,local_query_sigma,mix,local_fraction_pct,local_basin_fraction_pct,local_basin_queries,local_uniform_queries,adversarial_queries\n");
    fprintf(mf, "%llu,%u,%u,%u,%u,%.6f,%.6f,%s,%u,85,%u,%u,%u\n",
            (unsigned long long)seed, nv, nq, HITO9_OOD_DIM, k, sigma, DEFAULT_LOCAL_QUERY_SIGMA,
            mix == OOD_MIX_R ? "R" : mix == OOD_MIX_D ? "D" : "M50/50",
            hdr.local_fraction_percent, local_basin_count, local_uniform_count, adversarial_count);
    fclose(mf);

    printf("Hito 9 OOD dataset generated\n");
    printf("  file: %s\n  manifest: %s\n  seed: %llu\n  vectors: %u x %u\n  queries: %u\n",
           out_path, manifest_path, (unsigned long long)seed, nv, HITO9_OOD_DIM, nq);
    printf("  K: %u | sigma: %.2f | mix: %s\n", k, sigma,
           mix == OOD_MIX_R ? "R" : mix == OOD_MIX_D ? "D" : "M50/50");
    printf("  query labels: local=%u adversarial=%u\n",
           hdr.local_fraction_percent, 100U - hdr.local_fraction_percent);
    printf("  Ground Truth: not embedded; benchmark runner must compute exact eligible top-K.\n");

    free(centroids); free(vectors); free(categories); free(basin_nodes); free(queries);
    return 0;
}
