// Functions for evaluating or sampling from various probability distributions.

#include <RcppArmadillo.h>

const double log2pi = log(2.0 * M_PI);

double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logd) {
  double logl;
  int d = y.n_elem;
  arma::vec zvec(d);
  zvec = arma::chol(arma::inv(sigma)) * (y - mu);
  logl = as_scalar(-(log(arma::det(sigma)) + zvec.t() * zvec + d * log2pi) / 2);
  if (logd == false) {
    return exp(logl);
  }
  return logl;
}

arma::vec mvrnorm(arma::vec mu, arma::mat sigma) {
  int p = sigma.n_cols;
  return mu + arma::chol(sigma, "lower") * arma::randn(p);
}

arma::mat rwishart(int df, arma::mat S) {
  int d = S.n_rows;
  arma::vec z(d);
  arma::mat y(d, d, arma::fill::zeros);
  for (int i = 0; i < df; i++) {
    z = mvrnorm(arma::zeros(d), S);
    y = y + z * z.t();
  }
  return y;
}

arma::ivec randint(int n, int a, int b) {
  arma::ivec y(n);
  for (int i = 0; i < n; i++) {
    y(i) = floor(R::runif(0.0, 1.0) * (b - a + 1) + a);  
  }
  return y;
}