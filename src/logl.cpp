// Functions for evaluating various log-likelihood functions.

#include <RcppArmadillo.h>

double normlogl(arma::vec y, double mu, double sigma) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; ++i) {
    loglik = loglik + R::dnorm(y(i), mu, sigma, true);
  }
  return loglik;
}

double normlogl(arma::vec y, arma::vec mu, double sigma) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; ++i) {
    loglik = loglik + R::dnorm(y(i), mu(i), sigma, true);
  }
  return loglik;
}

double bernlogl(arma::vec y, arma::vec p) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; ++i) {
    loglik = loglik + R::dbinom(y(i), 1, p(i), true);
  }
  return loglik;
}

double bernlogl(arma::vec y, double p) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; ++i) {
    loglik = loglik + R::dbinom(y(i), 1, p, true);
  }
  return loglik;
}

double poislogl(arma::vec y, arma::vec lambda) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; ++i) {
    loglik = loglik + R::dpois(y(i), lambda(i), true); 
  }
  return loglik;
}

double poislogl(arma::vec y, double lambda) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; ++i) {
    loglik = loglik + R::dpois(y(i), lambda, true); 
  }
  return loglik;
}