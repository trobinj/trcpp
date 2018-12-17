// Functions for testing purposes only.

#include <RcppArmadillo.h>
#include "misc.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
void foo(arma::uvec x) {
  arma::mat y = expand(x);
  prnt(y);
}
