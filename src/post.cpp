// Source file for functions for sampling from a posterior distribution.

#include <RcppArmadillo.h>
#include "dist.h"

//' @export
// [[Rcpp::export]]
arma::vec meanpost(arma::mat y, arma::mat sigma, arma::vec mu0, arma::mat sigma0) {
  int n = y.n_rows;
  int m = y.n_cols;
  arma::mat sigma0inv(m,m);
  arma::mat sigmainv(m,n);
  arma::mat B(m,m);
  arma::vec b(m);
  sigma0inv = arma::inv(sigma0);
  sigmainv = arma::inv(sigma);
  B = arma::inv(sigma0inv + n * sigmainv);
  b = sigma0inv * mu0 + n * sigmainv * arma::mean(y).t();
  return mvrnorm(B * b, B);
}
