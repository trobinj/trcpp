// Miscellaneous specialized samplers.

#include <RcppArmadillo.h>

using namespace Rcpp;

arma::vec rnormsum(arma::vec mu, arma::vec sigma, double t, int n, double delta, bool rate) {
  int m = mu.n_elem;
  double z, num, den, s, a = 0.0;
  arma::vec y(m, arma::fill::zeros);
  for (int i = 1; i < n; ++i) {
    s = accu(y.head(m - 1));
    for (int j = 0; j < (m - 1); ++j) {
      z = R::rnorm(y(j), delta);
      num = R::dnorm(t - (s - y(j) + z), mu(m - 1), sigma(m - 1), true) + R::dnorm(z, mu(j), sigma(j), true);
      den = R::dnorm(t - s, mu(m - 1), sigma(m - 1), true) + R::dnorm(y(j), mu(j), sigma(j), true);
      if (R::runif(0.0, 1.0) < exp(num - den)) {
        y(j) = z;
        s = accu(y.head(m - 1));
        a++;
      }
    }
    y(m - 1) = t - accu(y.head(m - 1));
  }
  if (rate) {
    Rcout << "rnormsum transition rate: " << a / (n - 1) / (m - 1) << "\n";
  }
  return y;
}