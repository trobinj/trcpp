// Functions for testing purposes only.

#include <RcppArmadillo.h>
#include "dist.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
double foo(double mu, double sigma, double a, double b) {
  return rtnorm(mu, sigma, a, b);
}
