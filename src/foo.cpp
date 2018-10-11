// Functions for testing purposes only.

#include <RcppArmadillo.h>
#include "dist.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
void foo(int n, arma::vec mu, arma::mat sigma, bool lib) {
  arma::mat c(size(sigma));
  c = arma::chol(sigma);
  arma::vec y(size(mu));
  if (lib) {
    for (int i = 0; i < n; ++i) {
      y = mvrnorm(mu, c, TRUE);
    }
  } else {
    for (int i = 0; i < n; ++i) {
      y = arma::mvnrnd(mu, sigma); // slightly faster
    }
  }
}
