#include <RcppArmadillo.h>
#include "dist.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
List mprobit(arma::mat Y, arma::mat X, int samples) {
  int n = Y.n_rows;
  int m = Y.n_cols;
  int p = X.n_cols;
  
  arma::mat M(n, m);
  arma::mat B(p, m, arma::fill::zeros);
  arma::mat Z(n, m, arma::fill::zeros);
  
  arma::mat Bsave(samples, p * m);
  
  for (int t = 0; t < samples; t++) {
    
    M = X * B;
    
    // sample latent responses
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        Z(i, j) = 0.0;
      }
    }
  }
  
  return List::create(
    Named("beta") = wrap(Bsave)
  );
}