#include "../include/cost_model.h"
#include <cmath>
#include <limits>
extern "C" void cost_model_default_exact(ExactCostModel *m) {
 if (!m) return;
 /* Hito 7 owner-reported fit; calibrated for the current i5-13500 benchmark only. */
 m->exact_intercept_us=15.93; m->exact_per_eligible_us=0.00895;
 m->min_eligible_count=9000; m->max_eligible_count=26000; m->calibrated=true;
}
extern "C" double cost_model_exact_predict(const ExactCostModel *m,uint32_t n) {
 if(!m||!m->calibrated||n<m->min_eligible_count||n>m->max_eligible_count) return std::numeric_limits<double>::infinity();
 return m->exact_intercept_us+m->exact_per_eligible_us*n;
}
extern "C" EngineEstimate cost_model_hnsw_predict(const HnswCostModel *m,QueryRegime regime,double sel,uint32_t k) {
 EngineEstimate o{}; o.cost_us=std::numeric_limits<double>::infinity();
 if(!m||!m->points||!m->point_count||regime==QUERY_UNKNOWN||!std::isfinite(sel)) return o;
 const HnswCalibrationPoint *a=nullptr,*b=nullptr;
 for(size_t i=0;i<m->point_count;i++){const auto&p=m->points[i]; if(p.regime!=regime||!std::isfinite(p.p50_us))continue;
  if(p.selectivity_pct<=sel&&(!a||p.selectivity_pct>a->selectivity_pct))a=&p;
  if(p.selectivity_pct>=sel&&(!b||p.selectivity_pct<b->selectivity_pct))b=&p;}
 if(!a||!b)return o; /* No extrapolation outside calibration domain. */
 double t=a==b?0.0:(sel-a->selectivity_pct)/(b->selectivity_pct-a->selectivity_pct);
 o.cost_us=a->p50_us+t*(b->p50_us-a->p50_us);
 o.recall10=a->recall10+t*(b->recall10-a->recall10);
 o.recall100=a->recall100+t*(b->recall100-a->recall100);
 o.recall_lcb=a->recall10_lcb+t*(b->recall10_lcb-a->recall10_lcb);
 o.budget=t<0.5?a->ef_search:b->ef_search; o.supported=true; (void)k; return o;
}
