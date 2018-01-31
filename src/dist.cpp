// Functions for evaluating or sampling various probability distributions.

#include <RcppArmadillo.h>

const double log2pi = log(2.0 * M_PI);
const double logpi = log(M_PI);

double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logd) {
  int d = y.n_elem;
  double logl, lds, sign;
  log_det(lds, sign, sigma);
  arma::vec z(d);
  z = chol(inv(sigma)) * (y - mu);
  logl = as_scalar(-(lds + z.t() * z + d * log2pi) / 2);
  if (logd == false) {
    return exp(logl);
  }
  return logl;
}

arma::vec mvrnorm(arma::vec mu, arma::mat sigma) {
  int p = sigma.n_cols;
  return mu + arma::chol(sigma, "lower") * arma::randn(p);
}

arma::vec rmvt(arma::vec m, arma::mat s, double v) {
  return m + mvrnorm(arma::zeros(size(m)), s) / sqrt(R::rchisq(v) / v);
}

double dmvt(arma::vec y, arma::vec m, arma::mat s, double v, bool logd) {
  int p = y.n_elem;
  double t1, t2, t3, lds, sign;
  log_det(lds, sign, s);
  t1 = lgamma((v + p) / 2.0);
  t2 = lgamma(v / 2.0) + p * log(v) / 2.0 + p * logpi / 2.0 + lds / 2.0;
  t3 = -(v + p) / 2.0 * log(1.0 + as_scalar((y - m).t() * inv(s) * (y - m)) / v);
  if (logd) {
    return t1 - t2 + t3;
  }
  else {
    return exp(t1 - t2 + t3);
  }
}

// Multivariate gamma function \Gamma_p(a). 
double mvgamma(int p, double a, bool logd) {
  double y = 0.0;
  for (int j = 0; j < p; j++) {
    y = y + lgamma(a - j / 2.0);
  }
  y = y + logpi * p * (p - 1) / 4.0;
  if (logd) {
    return y;
  }
  else {
    return exp(y);
  }
}

double dwishart(arma::mat x, double n, arma::mat v, bool logd) {
  int p = x.n_rows;
  double y, sign, logdx, logdv;
  arma::log_det(logdx, sign, x);
  arma::log_det(logdv, sign, v);
  y = (n - p - 1) / 2.0 * logdx - trace(inv(v) * x) / 2.0
    - (n * p / 2.0 * log(2.0) + n / 2.0 * logdv + mvgamma(p, n / 2.0, true)); 
  if (logd) {
    return y;
  }
  else {
    return exp(y);
  }
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