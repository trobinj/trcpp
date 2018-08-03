// Functions for evaluating or sampling various probability distributions.

#include <RcppArmadillo.h>

const double log2pi = log(2.0 * M_PI);
const double logpi = log(M_PI);

// Sampler for a truncated positive (negative) normal random variable. This uses a rejection
// sampling algorithm proposed by Robert (1995, Statistics and Computing). 
double rtnormpos(double m, double s, bool pos) {
  double l, a, z, p, u; 
  l = pos ? -m/s : m/s;
  a = (l + sqrt(pow(l, 2) + 4.0)) / 2.0;
  do {
    z = R::rexp(1.0) / a + l;
    p = exp(-pow(z - a, 2) / 2.0);
    u = R::runif(0.0, 1.0);
  } while(u > p);
  return pos ? z * s + m : -z * s + m;
}

// Naive rejection sampler for truncated normal distribution. 
double rtnorm(double mu, double sigma, double a, double b) {
  double y;
  do {
    y = R::rnorm(mu, sigma);
  } while ((a > y) || (y > b));
  return y;
}

// Sampler for n random integers in [a,b].
arma::ivec randint(int n, int a, int b) {
  arma::ivec y(n);
  int c = b - a + 1;
  for (int i = 0; i < n; i++) {
    y(i) = floor(R::runif(0.0, 1.0) * c) + a;  
  }
  return y;
}

// Sample integer from 0 to n-1 with given sampling weights.
int rdiscrete(arma::vec wght) {
  int n = wght.n_elem;
  arma::vec prob = wght / accu(wght);
  double u = R::runif(0.0, 1.0);
  double cprb = 0.0;
  for (int y = 0; y < (n - 1); y++) {
    cprb = cprb + prob(y);
    if (u < cprb) {
      return y;
    } 
  }
  return n - 1; 
}

// Probability density function of a multivariate normal distribution.
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

// Sampler for multivariate normal distribution.
arma::vec mvrnorm(arma::vec mu, arma::mat sigma) {
  int p = sigma.n_cols;
  return mu + arma::chol(sigma, "lower") * arma::randn(p);
}

// Sampler for matrix-variate normal distribution.
arma::mat mvrnorm(arma::mat m, arma::mat u, arma::mat v) {
  arma::mat x = arma::randn(size(m));
  arma::mat a = arma::chol(u, "lower");
  arma::mat b = arma::chol(v, "upper");
  return m + a * x * b;
}

// Sampler for multivariate t distribution.
arma::vec rmvt(arma::vec m, arma::mat s, double v) {
  return m + mvrnorm(arma::zeros(size(m)), s) / sqrt(R::rchisq(v) / v);
}

// Probability density function of multivariate t distribution.
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

// Multivariate gamma function Gamma_p(a). 
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

// Probability density function of Wishart distribution.
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

// Sampler for Wishart distribution.
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

