#include "../include/alg_hnsw.h"
#include <windows.h>
#include <psapi.h>
#include <stdlib.h>

void memory_get_stats(MemoryStats *out) {
    if (!out) return;
    PROCESS_MEMORY_COUNTERS_EX pmc;
    ZeroMemory(&pmc, sizeof(pmc));
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
        out->working_set_bytes = pmc.WorkingSetSize;
        out->pagefile_bytes    = pmc.PagefileUsage;
        out->private_bytes     = pmc.PrivateUsage;
    } else {
        out->working_set_bytes = 0;
        out->pagefile_bytes    = 0;
        out->private_bytes     = 0;
    }
}

void *aligned_alloc_64(size_t bytes) {
    return _aligned_malloc(bytes, 64);
}

void aligned_free_64(void *ptr) {
    if (ptr) _aligned_free(ptr);
}
