// Miscellaneous specialized samplers.

#include <RcppArmadillo.h>

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
arma::mat rnormsum(arma::vec mu, arma::vec sigma, double t, int n, double delta) {
  int m = mu.n_elem;
  double z, num, den, tot;
  arma::mat y(n, m);
  y.row(0) = mu.t();
  for (int i = 1; i < n; i++) {
    y.row(i) = y.row(i-1);
    for (int j = 0; j < (m - 1); j++) {
      z = R::rnorm(y(i - 1, j), delta);
      tot = accu(y.submat(i, 0, i, m - 2));
      num = R::dnorm(t - (tot - y(i, j) + z), mu(m - 1), sigma(m - 1), true) + 
        R::dnorm(z, mu(j), sigma(j), true);
      den = R::dnorm(t - tot, mu(m - 1), sigma(m - 1), true) + 
        R::dnorm(y(i - 1, j), mu(j), sigma(j), true);
      if (R::runif(0.0, 1.0) < exp(num - den)) {
        y(i, j) = z;
      }
    }
    y(i, m - 1) = t - accu(y.submat(i, 0, i, m - 2));
  }
  return y;
}

// Note: Modify the above to not store all samples, and to maybe optionally report rejection rate.