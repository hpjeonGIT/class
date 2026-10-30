#include <Rcpp.h>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;
//' @export
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]
NumericVector sum_arrays(NumericVector x, NumericVector y) {
  #ifdef _OPENMP
  #pragma omp parallel
  #pragma omp master
  std::cout << "Number of OpenMP threads = " << omp_get_num_threads() << std::endl;
  #endif
  int n = x.size();
  NumericVector result(n);
  #pragma omp parallel default(private) shared(x,y,result,n)
  {
    #pragma omp for schedule(static)
    for(int i = 0; i < n; i++) {
      if (i%1000 == 0) {Rcpp:checkUserInterrupt();} // interrupt when the vector is too long
      result[i] = x[i] + y[i];
    }
  }
  return result;
}
