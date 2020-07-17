// Functions for evaluating or sampling various probability distributions.

#include <RcppArmadillo.h>
#include "misc.h"

// Sampler for sampling from the tail of a truncated normal distribution
// using either Marsaglia's (1964, Technometrics) rejection sampler or a
// simple rejection sampler, depending on the point of truncation.
double rnormtail(double a, double m, double s, bool pos) {
  double l, u, v, z;
  l = pos ? (a - m) / s : (m - a) / s;
  if (l > 0) {
    do {
      u = R::runif(0.0, 1.0);
      v = R::runif(0.0, 1.0);
      z = sqrt(pow(l,2) - 2.0 * log(u));
    } while(v < l / z);
  } else {
    do {
      z = R::rnorm(0.0, 1.0);
    } while (z < l);
  }
  return pos ? z * s + m : -z * s + m;
}

// Sampler for a truncated positive (or negative) normal random variable using
// a rejection sampling algorithm from Robert (1995, Statistics and Computing).
double rnormpos(double m, double s, bool pos) {
  double l, a, z, p, u;
  l = pos ? -m/s : m/s;
  a = (l + sqrt(pow(l,2) + 4.0)) / 2.0;
  do {
    z = R::rexp(1.0) / a + l;
    u = R::runif(0.0, 1.0);
    p = exp(-pow(z - a, 2) / 2.0);
  } while(u > p);
  return pos ? z * s + m : -z * s + m;
}

// Sampler for a finite interval-truncated normal random variable using a simple rejection algorithm.
double rnormrej(double m, double s, double a, double b) {
  double y;
  do {
    y = R::rnorm(m, s);
  } while (a > y || y > b);
  return y;
}

// Sampler for a finite interval-truncated normal random variable using
// a rejection algorithm from Robert (1995, Statistics and Computing).
// Includes an efficiency check against a simple rejection sampler.
double rnormint(double m, double s, double a, double b) {
  constexpr double sqrt2pi = sqrt(2 * M_PI);
  double low = (a - m) / s;
  double upp = (b - m) / s;
  double z, u, p, d;
  if (upp < 0) {
    d = pow(upp,2);
  } else if (low > 0) {
    d = pow(low,2);
  } else {
    d = 0.0;
  }
  if ((b - a) / d < sqrt2pi) {
    do {
      z = R::runif(low, upp);
      u = R::runif(0.0, 1.0);
      p = exp((d - pow(z,2)) / 2.0);
    } while (u > p);
  } else {
    do {
      z = R::rnorm(0.0, 1.0);
    } while (low > z || z > upp);
  }
  return z * s + m;
}

// Sampler for n random integers in [a,b].
arma::ivec randint(int n, int a, int b) {
  arma::ivec y(n);
  int c = b - a + 1;
  for (int i = 0; i < n; ++i) {
    y(i) = floor(R::runif(0.0, 1.0) * c) + a;
  }
  return y;
}

// Sampler for a random integer in [a,b].
int randint(int a, int b) {
  return floor(R::runif(0.0, 1.0) * (b - a + 1)) + a;
}

// Sample integer from 0 to n-1 with given sampling weights.
int rdiscrete(arma::vec wght) {
  int n = wght.n_elem;
  arma::vec prob = wght / accu(wght);
  double u = R::runif(0.0, 1.0);
  double cprb = 0.0;
  for (int y = 0; y < n - 1; ++y) {
    cprb = cprb + prob(y);
    if (u < cprb) {
      return y;
    }
  }
  return n - 1;
}

// Fisher-Yates random shuffle algorithm for vec class.
void shuffle(arma::vec & x) {
  int j;
  int n = x.n_elem;
  for (int i = 0; i < n - 1; ++i) {
    j = randint(i, n - 1);
    vswap(x, i, j);
  }
}

// Simple random sampling from vector x using the Fisher-Yates shuffle algorithm.
arma::vec srs(arma::vec x, int n) {
  int j;
  int l = x.n_elem;
  for (int i = 0; i < n; ++i) {
    j = randint(i, l - 1);
    vswap(x, i, j);
  }
  return x.head(n);
}

// Simple random sampling from integers 0,1,...,m-1 using the Fisher-Yates algorithm.
arma::vec srs(int m, int n) {
  int j;
  arma::vec x = arma::regspace(0, m - 1);
  for (int i = 0; i < n; ++i) {
    j = randint(i, m - 1);
    vswap(x, i, j);
  }
  return x.head(n);
}

// Probability density function of a multivariate normal distribution.
double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logd) {
  constexpr double log2pi = log(2.0 * M_PI);
  int d = y.n_elem;
  double lnll, lds, sign;
  log_det(lds, sign, sigma);
  arma::vec z(d);
  z = chol(inv(sigma)) * (y - mu);
  lnll = as_scalar(-(lds + z.t() * z + d * log2pi) / 2);
  return logd ? lnll : exp(lnll);
}

// Sampler for multivariate normal distribution.
arma::vec mvrnorm(arma::vec mu, arma::mat sigma, bool cholesky) {
  int p = sigma.n_cols;
  if (cholesky) {
    return mu + sigma * arma::randn(p);
  }
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
  return m + mvrnorm(arma::zeros(size(m)), s, false) / sqrt(R::rchisq(v) / v);
}

// Probability density function of multivariate t distribution.
double dmvt(arma::vec y, arma::vec m, arma::mat s, double v, bool logd) {
  constexpr double logpi = log(M_PI);
  int p = y.n_elem;
  double t1, t2, t3, lds, sign;
  log_det(lds, sign, s);
  t1 = lgamma((v + p) / 2.0);
  t2 = lgamma(v / 2.0) + p * log(v) / 2.0 + p * logpi / 2.0 + lds / 2.0;
  t3 = -(v + p) / 2.0 * log(1.0 + as_scalar((y - m).t() * inv(s) * (y - m)) / v);
  return logd ? t1 - t2 + t3 : exp(t1 - t2 + t3);
}

// Multivariate gamma function Gamma_p(a).
double mvgamma(int p, double a, bool logd) {
  constexpr double logpi = log(M_PI);
  double y = 0.0;
  for (int j = 0; j < p; ++j) {
    y = y + lgamma(a - j / 2.0);
  }
  y = y + logpi * p * (p - 1) / 4.0;
  return logd ? y : exp(y);
}

// Probability density function of a Wishart distribution with scale matrix v and degrees of freedom n.
double dwishart(arma::mat x, double n, arma::mat v, bool logd) {
  constexpr double log2 = log(2.0);
  int p = x.n_rows;
  double y, sign, logdx, logdv;
  arma::log_det(logdx, sign, x);
  arma::log_det(logdv, sign, v);
  y = (n - p - 1) / 2.0 * logdx - trace(inv(v) * x) / 2.0
    - (n * p / 2.0 * log2 + n / 2.0 * logdv
    + mvgamma(p, n / 2.0, true));
  return logd ? y : exp(y);
}

// Sampler for Wishart distribution.
arma::mat rwishart(int df, arma::mat S) {
  int d = S.n_rows;
  arma::vec z(d);
  arma::mat y(d, d, arma::fill::zeros);
  arma::mat C = arma::chol(S, "lower");
  for (int i = 0; i < df; ++i) {
    z = mvrnorm(arma::zeros(d), C, true);
    y = y + z * z.t();
  }
  return y;
}

// Geweke-Hajivassiliou-Keane (GHK) importance sampling algorithm.
namespace ghkspc {
    double pnorm(double z) {
        return R::pnorm(z, 0.0, 1.0, true, false);
    }
}
double ghk(arma::vec m, arma::mat s, arma::vec low, arma::vec upp, int n) {
    using namespace ghkspc;

    d = m.n_elem;
    arma::vec lw = low - m;
    arma::vec up = upp - m;
    arma::vec q(d, arma::fill::zeros);
    arma::vec u(d - 1, arma::fill::zeros);
    arma::mat C = arma::chol(sigma, "lower");
    double v, l, u, prb = 0.0;

    for (int i = 0; i < n; ++i) {

        q(0) = pnorm(up(0) / C(0,0)) - pnorm(lw(0) / C(0,0));
        u(0) = rnormint(0.0, 1.0, lw(0) / C(0,0), up(0) / C(0,0));

        for (int j = 1; j < d; ++j) {
            v = as_scalar(C(j,arma::span(0,j-1)) * u.head(j-1));
            l = (lw(j) - v) / C(j,j);
            u = (up(j) - v) / C(j,j);
            q(j) = pnorm(u) - pnorm(l);
            if (j < d - 1) {
                u(j) = rnormint(0.0, 1.0, l, u);
            }
        }
        prb = prb + prod(q);
    }

    return prb / n;
}

/* The pmvnorm function uses a Monte Carlo algorithm due to Genz (1992, Journal of
 * Computational and Graphical Statistics). I added a parameter nmin for the
 * minimum number of iterations before checking the estimated Monte Carlo error.
 */

// namespace pmvnormspc {
// double pnorm(double z) {
//   return R::pnorm(z, 0.0, 1.0, true, false);
// }
// double qnorm(double p) {
//   return R::qnorm(p, 0.0, 1.0, true, false);
// }
// }
//
// double pmvnorm(arma::vec a, arma::vec b, arma::vec mu, arma::mat sigma, double epsilon, double alpha, int nmin, int nmax) {
//
//   using namespace pmvnormspc;
//
//   arma::mat c = chol(sigma, "lower");
//   a = a - mu;
//   b = b - mu;
//
//   double intsum = 0.0;
//   double varsum = 0.0;
//   double interr = 0.0;
//   double cy;
//   int m = a.n_elem;
//   int N;
//
//   arma::vec y(m - 1);
//   arma::vec w(m - 1);
//   arma::vec d(m); d(0) = pnorm(a(0) / c(0,0));
//   arma::vec e(m); e(0) = pnorm(b(0) / c(0,0));
//   arma::vec f(m); f(0) = e(0) - d(0);
//
//   for (int k = 0; k < nmax; ++k) {
//     w.randu();
//     for (int i = 1; i < m; ++i) {
//       y(i - 1) = qnorm(d(i - 1) + w(i - 1) * (e(i - 1) - d(i - 1)));
//       cy = 0.0;
//       for (int j = 0; j < i; ++j) {
//         cy = cy + c(i, j) * y(j);
//       }
//       d(i) = pnorm((a(i) - cy) / c(i, i));
//       e(i) = pnorm((b(i) - cy) / c(i, i));
//       f(i) = (e(i) - d(i)) * f(i - 1);
//     }
//     N = k + 1;
//     intsum = intsum + f(m - 1);
//     varsum = varsum + pow(f(m - 1), 2);
//
//     if (N > nmin) {
//       interr = alpha * sqrt((varsum / N - pow(intsum / N, 2)) / N);
//       if (interr < epsilon) {
//         break;
//       }
//     }
//   }
//
//   return intsum / N;
// }
