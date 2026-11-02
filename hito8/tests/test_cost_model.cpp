#include "../include/cost_model.h"
#include <cassert>
#include <cmath>
#include <limits>
int main(){ExactCostModel e{};cost_model_default_exact(&e);assert(std::fabs(cost_model_exact_predict(&e,10000)-105.43)<0.02);assert(std::isinf(cost_model_exact_predict(&e,1000)));
HnswCalibrationPoint p[]={{QUERY_LOCAL,10,50,160,.97,.90,.96,.0,5,5000},{QUERY_LOCAL,15,50,122,.959,.88,.951,.0,5,5000}};
HnswCostModel m{p,2};auto x=cost_model_hnsw_predict(&m,QUERY_LOCAL,12.5,10);assert(x.supported&&std::fabs(x.cost_us-141)<.001);assert(!cost_model_hnsw_predict(&m,QUERY_ADVERSARIAL,12.5,10).supported);assert(!cost_model_hnsw_predict(&m,QUERY_LOCAL,5,10).supported);return 0;}
