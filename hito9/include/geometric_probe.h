#ifndef HITO9_GEOMETRIC_PROBE_H
#define HITO9_GEOMETRIC_PROBE_H

#include "../../hito8/include/hybrid_engine.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEOMETRIC_PROBE_MAX_PROTOTYPES 128U

typedef struct {
    uint32_t num_prototypes;
    uint32_t dim;
    float local_threshold_sq;       /* d^2 < local_threshold_sq -> QUERY_LOCAL */
    float adversarial_threshold_sq; /* d^2 > adversarial_threshold_sq -> QUERY_ADVERSARIAL */
    /* If local_threshold_sq <= d^2 <= adversarial_threshold_sq -> QUERY_UNKNOWN */
    uint64_t train_seed;
    uint32_t train_sample_size;
    float prototypes[GEOMETRIC_PROBE_MAX_PROTOTYPES * 64U];
} GeometricProbe;

typedef struct {
    uint32_t tp; /* R correctly classified as R */
    uint32_t fn; /* R misclassified as D */
    uint32_t fp; /* D misclassified as R */
    uint32_t tn; /* D correctly classified as D */
    uint32_t unknown_r; /* R classified as UNKNOWN */
    uint32_t unknown_d; /* D classified as UNKNOWN */
    uint32_t total_r;
    uint32_t total_d;
    double precision;
    double recall;
    double specificity;
    double fpr;
    double fnr;
    double balanced_accuracy;
    double avg_latency_ns;
} ConfusionMatrix;

/* Build probe using spherical k-means on training vectors */
int geometric_probe_build(
    GeometricProbe *probe,
    const float *train_vectors,
    uint32_t num_vectors,
    uint32_t dim,
    uint32_t num_prototypes,
    uint32_t num_iterations,
    uint64_t seed
);

/* Classify a single query into QUERY_LOCAL, QUERY_ADVERSARIAL, or QUERY_UNKNOWN */
QueryRegime geometric_probe_classify(
    const GeometricProbe *probe,
    const float *query,
    float *out_min_dist_sq
);

/* Compute full confusion matrix and classification metrics */
void confusion_matrix_compute(
    const GeometricProbe *probe,
    const float *queries_R,
    uint32_t nq_R,
    const float *queries_D,
    uint32_t nq_D,
    uint32_t dim,
    ConfusionMatrix *cm
);

/* Persistence */
int geometric_probe_save(const GeometricProbe *probe, const char *filepath);
int geometric_probe_load(GeometricProbe *probe, const char *filepath);

#ifdef __cplusplus
}
#endif

#endif /* HITO9_GEOMETRIC_PROBE_H */
