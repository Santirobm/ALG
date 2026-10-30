#include "heap.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int g_hnsw_use_heaps = 0;

int distpair_compare_asc(const DistPair *a, const DistPair *b) {
    const int an = isnan(a->dist), bn = isnan(b->dist);
    if (an != bn) return an ? 1 : -1; /* NaN siempre al final */
    if (!an) {
        if (a->dist < b->dist) return -1;
        if (a->dist > b->dist) return 1;
    }
    return (a->id > b->id) - (a->id < b->id);
}
static int qsort_asc(const void *a, const void *b) {
    return distpair_compare_asc((const DistPair *)a, (const DistPair *)b);
}
static void min_sift_up(DistPair *a, uint32_t i) {
    while (i) {
        uint32_t p = (i - 1U) >> 1U;
        if (distpair_compare_asc(&a[p], &a[i]) <= 0) break;
        DistPair t=a[p]; a[p]=a[i]; a[i]=t; i=p;
    }
}
static void min_sift_down(DistPair *a, uint32_t n, uint32_t i) {
    for (;;) {
        uint32_t l=i*2U+1U, r=l+1U, m=i;
        if (l<n && distpair_compare_asc(&a[l],&a[m])<0) m=l;
        if (r<n && distpair_compare_asc(&a[r],&a[m])<0) m=r;
        if (m==i) break;
        DistPair t=a[i]; a[i]=a[m]; a[m]=t; i=m;
    }
}
static void max_sift_up(DistPair *a, uint32_t i) {
    while (i) {
        uint32_t p=(i-1U)>>1U;
        if (distpair_compare_asc(&a[p],&a[i])>=0) break;
        DistPair t=a[p]; a[p]=a[i]; a[i]=t; i=p;
    }
}
static void max_sift_down(DistPair *a, uint32_t n, uint32_t i) {
    for (;;) {
        uint32_t l=i*2U+1U, r=l+1U, m=i;
        if (l<n && distpair_compare_asc(&a[l],&a[m])>0) m=l;
        if (r<n && distpair_compare_asc(&a[r],&a[m])>0) m=r;
        if (m==i) break;
        DistPair t=a[i]; a[i]=a[m]; a[m]=t; i=m;
    }
}
void hnsw_candidates_push(DistPair *a, uint32_t *n, uint32_t cap, DistPair v) {
    if (*n>=cap) return;
    if (!g_hnsw_use_heaps) { a[(*n)++]=v; return; }
    a[*n]=v; min_sift_up(a,*n); (*n)++;
}
DistPair hnsw_candidates_pop(DistPair *a, uint32_t *n) {
    if (!g_hnsw_use_heaps) {
        qsort(a,*n,sizeof(*a),qsort_asc);
        DistPair v=a[0]; if (*n>1) memmove(a,a+1,(*n-1U)*sizeof(*a)); --*n; return v;
    }
    DistPair v=a[0]; --*n;
    if (*n) { a[0]=a[*n]; min_sift_down(a,*n,0); }
    return v;
}
DistPair hnsw_results_worst(DistPair *a, uint32_t n) {
    if (g_hnsw_use_heaps) return a[0];
    qsort(a,n,sizeof(*a),qsort_asc);
    return a[n-1U];
}
void hnsw_results_insert(DistPair *a, uint32_t *n, uint32_t cap, DistPair v) {
    if (*n<cap) {
        if (!g_hnsw_use_heaps) { a[(*n)++]=v; return; }
        a[*n]=v; max_sift_up(a,*n); (*n)++; return;
    }
    if (cap==0) return;
    if (distpair_compare_asc(&v,&a[g_hnsw_use_heaps ? 0 : (*n-1U)]) >= 0) return;
    if (!g_hnsw_use_heaps) { a[*n-1U]=v; return; }
    a[0]=v; max_sift_down(a,*n,0);
}
void hnsw_results_sort(DistPair *a, uint32_t n) {
    qsort(a,n,sizeof(*a),qsort_asc);
}
