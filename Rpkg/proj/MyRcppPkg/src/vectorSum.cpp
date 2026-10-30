#include <Rcpp.h>
using namespace Rcpp;

//' @export
// [[Rcpp::export]]
NumericVector sum_arrays(NumericVector x, NumericVector y) {
  int n = x.size();
  NumericVector result(n);
  for(int i = 0; i < n; i++) {
    if (i%1000 == 0) {Rcpp:checkUserInterrupt();} // interrupt when the vector is too long
    result[i] = x[i] + y[i];
  }
  return result;
}
