// Functions for testing purposes only.

#include <RcppArmadillo.h>
#include "gslf.h"

//' @export
// [[Rcpp::export]]
arma::mat foo(int n) {
  arma::mat y(n,2);
  arma::vec node(n);
  arma::vec wght(n);
  ghquad(n, node, wght);
  for (int i = 0; i < n; ++i) {
    y(i,0) = node(i);
    y(i,1) = wght(i);
  }
  return y;
}
