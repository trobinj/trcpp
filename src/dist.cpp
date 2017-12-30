// Functions for evaluating or sampling from various probability distributions.

#include <RcppArmadillo.h>

const double log2pi = log(2.0 * M_PI);

double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logdensity) {
  double logl;
  int d = y.n_elem;
  arma::vec zvec(d);
  zvec = arma::chol(arma::inv(sigma)) * (y - mu);
  logl = as_scalar(-(log(arma::det(sigma)) + zvec.t() * zvec + d * log2pi) / 2);
  if (logdensity == false) {
    return exp(logl);
  }
  return logl;
}

arma::vec mvrnorm(arma::vec mu, arma::mat sigma) {
  int p = sigma.n_cols;
  return mu + arma::chol(sigma, "lower") * arma::randn(p);
}