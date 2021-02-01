// Functions for testing purposes only.

#include <RcppArmadillo.h>
#include "dist.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
double foo(arma::vec m, arma::mat s, arma::vec low, arma::vec upp, int n) {
  return ghk(m, s, low, upp, n );
}