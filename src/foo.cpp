// Functions for testing purposes only.

#include <RcppArmadillo.h>
#include "misc.h"
#include "comb.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
void foo(List x) {
  arma::mat y = x["x"];
  Rcout << y << "\n";
}
