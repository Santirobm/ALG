#include "../include/hybrid_engine.h"
#include "../include/cost_model.h"
#include <limits>
extern "C" int hybrid_search(const HybridEngineBindings*b,const QueryFeatures*f,const QualityRequirements*r,
 const ExactCostModel*em,const HnswCostModel*hm,const float*q,const void*pred,HybridSearchOutput*out,HybridQueryResult*meta){
 if(!b||!f||!r||!q||!out||!meta||!b->exact_scan.available||!b->exact_scan.search||!r->k||out->capacity<r->k)return -1;
 *meta={}; double ec=cost_model_exact_predict(em,f->eligible_count);
 EngineEstimate h=cost_model_hnsw_predict(hm,f->regime,f->selectivity_pct,r->k);
 bool hav=b->hnswlib.available&&b->hnswlib.search&&h.supported;
 HybridEngineId id=ENGINE_EXACT_SCAN; uint32_t budget=0; double chosen=ec;
 if(r->policy==POLICY_BALANCED){if(hav&&h.recall_lcb>=r->target_recall&&h.cost_us<ec){id=ENGINE_HNSWLIB;budget=h.budget;chosen=h.cost_us;}}
 else if(r->policy==POLICY_APPROXIMATE){if(hav&&h.cost_us<ec){id=ENGINE_HNSWLIB;budget=h.budget;chosen=h.cost_us;}}
 else if(r->policy!=POLICY_EXACT)return -2;
 if(r->max_latency_budget_us>0&&chosen>r->max_latency_budget_us&&id!=ENGINE_EXACT_SCAN){id=ENGINE_EXACT_SCAN;budget=0;chosen=ec;meta->fallback_triggered=true;}
 const HybridEngineBinding*e=id==ENGINE_HNSWLIB?&b->hnswlib:&b->exact_scan;
 int rc=e->search(e->context,q,pred,r->k,budget,out);
 if(rc!=0&&id!=ENGINE_EXACT_SCAN){meta->fallback_triggered=true;id=ENGINE_EXACT_SCAN;rc=b->exact_scan.search(b->exact_scan.context,q,pred,r->k,0,out);chosen=ec;}
 meta->chosen_engine=id;meta->estimated_cost_us=chosen;meta->estimated_recall=id==ENGINE_HNSWLIB?h.recall10:1.0;
 meta->recall_lower_confidence_bound=id==ENGINE_HNSWLIB?h.recall_lcb:1.0;meta->quality_supported=id==ENGINE_EXACT_SCAN||h.supported;
 meta->actual_latency_us=std::numeric_limits<double>::quiet_NaN();meta->regret_us=std::numeric_limits<double>::quiet_NaN();return rc;
}
