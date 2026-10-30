#include "../include/alg_hnsw.h"
#include "../include/heap.h"
#include "../include/sq8.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_double(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double recall_k(const SearchResult *p,const SearchResult *g,uint32_t k){
    uint32_t n=g->returned_count<k?g->returned_count:k, m=0, pn=p->returned_count<k?p->returned_count:k;
    if(!n)return 1.0;
    for(uint32_t i=0;i<pn;i++)for(uint32_t j=0;j<n;j++)if(p->ids[i]==g->ids[j]){m++;break;}
    return (double)m/(double)n;
}
static void run_case(uint32_t seed,const char *scenario,const char *name,int sq8_mode,HnswIndex *index,const Sq8Index *sq8,
                     const AttributeSet *attrs,const FilterPredicate *pred,const float *queries,
                     const SearchResult *gt,uint32_t nq,uint32_t ef,FILE *csv){
    double *lat=(double*)malloc((size_t)nq*sizeof(double));
    double r10=0,r100=0; uint64_t nav=0,rerank=0,visited=0;
    LARGE_INTEGER fq,t0,t1; QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&t0);
    for(uint32_t q=0;q<nq;q++){
        LARGE_INTEGER a,b; QueryPerformanceCounter(&a);
        SearchResult r=sq8_mode?hnsw_search_h2_sq8_filteraware(index,sq8,&queries[(size_t)q*index->dim],attrs,pred,100,ef)
                               :hnsw_search_h2_filteraware(index,&queries[(size_t)q*index->dim],attrs,pred,100,ef);
        QueryPerformanceCounter(&b);
        lat[q]=(double)(b.QuadPart-a.QuadPart)*1e6/(double)fq.QuadPart;
        r10+=recall_k(&r,&gt[q],10); r100+=recall_k(&r,&gt[q],100);
        nav+=r.distance_comps; rerank+=r.rerank_comps; visited+=r.visited_nodes;
    }
    QueryPerformanceCounter(&t1); qsort(lat,nq,sizeof(double),cmp_double);
    double sec=(double)(t1.QuadPart-t0.QuadPart)/(double)fq.QuadPart;
    double p50=lat[(size_t)(nq*.50)],p95=lat[(size_t)(nq*.95)],p99=lat[(size_t)(nq*.99)];
    double qps=sec>0?nq/sec:0;
    printf("%-12s R10 %.4f R100 %.4f p50 %.2f us p95 %.2f us p99 %.2f us QPS %.1f nav_dist %.1f rerank %.1f visited %.1f\n",
        name,r10/nq,r100/nq,p50,p95,p99,qps,(double)nav/nq,(double)rerank/nq,(double)visited/nq);
    if(csv)fprintf(csv,"%u,%s,%s,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",seed,scenario,name,r10/nq,r100/nq,p50,p95,p99,qps,(double)nav/nq,(double)rerank/nq,(double)visited/nq);
    free(lat);
}
int main(int argc,char **argv){
    SetConsoleOutputCP(65001); setvbuf(stdout,NULL,_IONBF,0);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS); SetThreadAffinityMask(GetCurrentThread(),1ULL);
    uint32_t nv=100000,nq=1000,ef=100,sel=5;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--num-vectors")&&i+1<argc)nv=(uint32_t)atoi(argv[++i]);
        else if(!strcmp(argv[i],"--num-queries")&&i+1<argc)nq=(uint32_t)atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ef-search")&&i+1<argc)ef=(uint32_t)atoi(argv[++i]);
        else if(!strcmp(argv[i],"--selectivity")&&i+1<argc)sel=(uint32_t)atoi(argv[++i]);
    }
    if(!nv||!nq||!ef||ef>512||sel>100){fprintf(stderr,"Argumentos fuera de rango.\n");return 2;}
    const uint64_t seeds[5]={42ULL,1337ULL,987654321ULL,0xCAFEBABEULL,0xDEADBEEFULL};
    CreateDirectoryA("results",NULL); FILE *csv=fopen("results/sq8_ablation.csv","w");
    if(csv)fprintf(csv,"Seed,Scenario,Variant,Recall10,Recall100,p50_us,p95_us,p99_us,QPS,NavDistanceComps,RerankComps,Visited\n");
    FilterPredicate pred=filter_create_predicate(sel);
    printf("SQ8 experiment | N=%u Q=%u selectivity=%u%% efSearch=%u | 5 seeds, R/D\n",nv,nq,sel,ef);
    for(uint32_t si=0;si<5;si++){
        printf("\n=== Seed %u ===\n",si+1);
        float *vectors=vectors_generate_synthetic(nv,VECTOR_DIM,seeds[si]);
        AttributeSet *attrs=attributes_generate(nv,seeds[si]);
        HnswIndex *index=hnsw_create(nv,VECTOR_DIM,HNSW_M_DEFAULT,HNSW_EF_CONSTRUCT);
        if(!vectors||!attrs||!index){fprintf(stderr,"Fallo de asignacion.\n");return 3;}
        LARGE_INTEGER fq,a,b;QueryPerformanceFrequency(&fq);QueryPerformanceCounter(&a);
        for(uint32_t i=0;i<nv;i++)hnsw_insert(index,i,&vectors[(size_t)i*VECTOR_DIM]);
        QueryPerformanceCounter(&b);printf("Build HNSW %.2f s\n",(double)(b.QuadPart-a.QuadPart)/(double)fq.QuadPart);
        QueryPerformanceCounter(&a); Sq8Index *sq8=sq8_build(vectors,nv,VECTOR_DIM); QueryPerformanceCounter(&b);
        if(!sq8){fprintf(stderr,"Fallo construyendo SQ8.\n");return 4;}
        printf("SQ8 build %.2f s | payload %.2f MiB (float vectors %.2f MiB) | total resident payload still includes float32 originals for rerank\n",
            (double)(b.QuadPart-a.QuadPart)/(double)fq.QuadPart,(double)sq8->bytes/1048576.0,(double)((size_t)nv*VECTOR_DIM*sizeof(float))/1048576.0);
        for(int sc=0;sc<2;sc++){
            bool reuse=(sc==0); const char *sn=reuse?"R_Local":"D_Adversarial";
            printf("\n-- %s --\n",sn);
            float *queries=queries_generate_scenario(nq,VECTOR_DIM,vectors,nv,reuse,seeds[si]^0xABCDULL);
            SearchResult *gt=(SearchResult*)malloc((size_t)nq*sizeof(SearchResult));
            if(!queries||!gt){fprintf(stderr,"Fallo de memoria consultas.\n");return 5;}
            for(uint32_t q=0;q<nq;q++)gt[q]=ground_truth_linear_scan(vectors,nv,VECTOR_DIM,attrs,&pred,&queries[(size_t)q*VECTOR_DIM],100);
            g_hnsw_use_heaps=1;
            run_case(si+1,sn,"H2_F32_Heap",0,index,sq8,attrs,&pred,queries,gt,nq,ef,csv);
            run_case(si+1,sn,"H2_SQ8_Heap",1,index,sq8,attrs,&pred,queries,gt,nq,ef,csv);
            aligned_free_64(queries);free(gt);
        }
        sq8_free(sq8);hnsw_free(index);attributes_free(attrs);aligned_free_64(vectors);
    }
    if (csv) fclose(csv);
    return 0;
}
