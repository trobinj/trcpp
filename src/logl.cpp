// Functions for evaluating a log-likelihood.

#include <RcppArmadillo.h>

const double log2pi = log(2.0 * M_PI);

/*
double lmerlogl(arma::vec y, arma::mat x, arma::mat z, int m, 
  arma::vec beta, arma::mat phiv, double psiv) {
  double logl = 0.0, lds, sign;
  int n = y.n_elem/m;
  int p = x.n_cols;
  int q = z.n_cols;
  arma::vec yi(m);
  arma::mat xi(m,p);
  arma::mat zi(m,q);
  arma::mat si(m,m);
  for (int i = 0; i < n; i++) {
    yi = y(arma::span(m * i, m * i + m - 1));
    xi = x.rows(m * i, m * i + m - 1);
    zi = z.rows(m * i, m * i + m - 1);
    si = zi * phiv * zi.t() + psiv * arma::eye(m,m);
    log_det(lds, sign, si);
    logl = logl + lds + as_scalar((yi - xi*beta).t() * inv(si) * (yi - xi*beta));
  }
  return -(logl + n * log2pi * m)/2;
}
*/

double normlogl(arma::vec y, arma::vec mu, double sigma) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; i++) {
    loglik = loglik + R::dnorm(y(i), mu(i), sigma, true);
  }
  return loglik;
}

double bernlogl(arma::vec y, arma::vec p) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; i++) {
    loglik = loglik + R::dbinom(y(i), 1, p(i), true);
  }
  return loglik;
}

double poislogl(arma::vec y, arma::vec lambda) {
  int n = y.n_elem;
  double loglik = 0.0;
  for (int i = 0; i < n; i++) {
    loglik = loglik + R::dpois(y(i), lambda(i), true); 
  }
  return loglik;
}