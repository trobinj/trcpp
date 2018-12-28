// Functions for testing purposes only.

#include <RcppArmadillo.h>
#include "misc.h"
#include "comb.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
void foo(int n) {
  arma::vec node(n);
  arma::vec wght(n);
  ghquad(n, node, wght);
  prnt(node);
  prnt(wght);
}
