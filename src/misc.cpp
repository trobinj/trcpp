#include "RcppArmadillo.h"

//' @export
// [[Rcpp::export]]
arma::vec lowertri(arma::mat x) {
  int n = x.n_rows;
  int m = x.n_cols;
  int d = std::min(n,m) * (std::min(n,m) + 1) / 2;
  if (n > m) {
    d = d + (n - m) * m;
  }
  arma::vec y(d);
  int t = 0;
  for (int j = 0; j < std::min(n,m); j++) {
    for (int i = j; i < n; i++) {
      y(t) = x(i,j);
      t = t + 1;
    }
  }
  return y;
}
