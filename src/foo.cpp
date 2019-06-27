// Functions for testing purposes only.

#include <RcppArmadillo.h>
#include "dist.h"

using namespace Rcpp;

double trnorm(double m, double s, double a, double b) {
  double y;
  do {
    y = R::rnorm(m, s);
  } while (a > y || y > b);
  return y;
}

//' @export
// [[Rcpp::export]]
void foo(int n, double m, double s, double a, double b) {
  arma::mat y(n, 2);
  for (int i = 0; i < n; ++i) {
    y(i,0) = trnorm(m, s, a, b);
    y(i,1) = rnormint(m, s, a, b);
  }
  Rcpp::Rcout << mean(y,0) << "\n";
  Rcpp::Rcout << stddev(y,0) << "\n";
}
