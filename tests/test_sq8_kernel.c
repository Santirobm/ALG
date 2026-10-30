#include "../include/sq8.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

int main(void) {
    const uint32_t n = 1000, dim = 64;
    float *vectors = (float *)malloc((size_t)n * dim * sizeof(float));
    if (!vectors) return 2;
    for (uint32_t i = 0; i < n * dim; ++i)
        vectors[i] = sinf((float)i * 0.017f) * 3.7f + (float)(i % 13) * 0.11f;
    Sq8Index *sq8 = sq8_build(vectors, n, dim);
    if (!sq8) return 3;
    uint8_t q[64];
    sq8_encode_query(sq8, vectors + 37U * dim, q);
    float max_abs_error = 0.0f;
    for (uint32_t id = 0; id < n; ++id) {
        float got = l2_distance_sq8_avx2(sq8, q, id), ref = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            float delta = ((int)q[d] - (int)sq8->codes[(size_t)id * dim + d]) * sq8->scales[d];
            ref += delta * delta;
        }
        float err = fabsf(got - ref);
        if (err > max_abs_error) max_abs_error = err;
    }
    printf("max_abs_error=%.9g aligned64=%s\n", max_abs_error,
           ((uintptr_t)sq8->codes % 64U) == 0U ? "PASS" : "FAIL");
    int ok = max_abs_error < 1e-3f && ((uintptr_t)sq8->codes % 64U) == 0U;
    sq8_free(sq8); free(vectors);
    return ok ? 0 : 1;
}
