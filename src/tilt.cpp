// Functions for tilted rejection sampling.

#include <RcppArmadillo.h>
#include "dist.h" // for rdiscrete

arma::vec multscor(arma::mat prb, arma::vec t, arma::vec s) {
  int n = prb.n_rows;
  int m = prb.n_cols;
  arma::vec y(m, arma::fill::zeros);
  arma::vec temp(m);
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      y(j) = y(j) + prb(i,j) * exp(t(j)) / as_scalar(prb.row(i) * exp(t));
    }
  }
  return s - y;
}

arma::mat multjcbn(arma::mat prb, arma::vec t, arma::vec s) {
  int m = prb.n_cols;
  arma::mat J(m, m);
  arma::vec d(m, arma::fill::zeros);
  for (int j = 0; j < m; ++j) {
    d(j) = 0.0001;
    J.col(j) = (multscor(prb, t + d, s) - multscor(prb, t - d, s)) / (2 * d(j));
    d(j) = 0.0;
  }
  return J;
}

// Need to add more sophisticated iteration control here.
arma::vec multroot(arma::mat prb, arma::vec s) {
  int m = prb.n_cols;
  arma::vec y(m, arma::fill::zeros);
  for (int i = 0; i < 10; ++i) {
    y.tail(m-1) = y.tail(m-1) - inv(multjcbn(prb, y, s).submat(1, 1, m-1, m-1)) * multscor(prb, y, s).tail(m-1);
  }
  return y;
}

arma::mat multrjct(arma::mat prb, arma::vec s) {
  int n = prb.n_rows;
  int m = prb.n_cols;
  arma::mat y(n, m);
  arma::mat p(n, m);
  arma::vec num(m);
  double den;
  arma::vec t = multroot(prb, s);
  for (int i = 0; i < n; ++i) {
    num = prb.row(i).t() % exp(t);
    den = accu(num);
    for (int j = 0; j < m; ++j) {
      prb(i, j) = num(j) / den;
    }
  }
  do {
    y.fill(0);
    for (int i = 0; i < n; ++i) {
      y(i, rdiscrete(prb.row(i).t())) = 1; // note: maybe transpose prb first to speed this up
    }
  } while (any(sum(y, 0).t() != s));
  return y * arma::regspace(0, m - 1);
}

double bernscor(arma::vec prb, double t, int s) {
  int n = prb.n_elem;
  double y = 0.0;
  for (int i = 0; i < n; ++i) {
    y = y + prb(i) * exp(t) / (prb(i) * exp(t) + 1 - prb(i));
  }
  return s - y;
}

double bernroot(arma::vec prb, int s, double a, double b, int n) {
  double fa, fb, c = 0.0;
  for (int i = 0; i < n; ++i) {
    fb = bernscor(prb, b, s);
    fa = bernscor(prb, a, s);
    if (fb == fa) {
      break;
    }
    c = b - fb * (b - a) / (fb - fa);
    a = b;
    b = c;
  }
  return c;
}

arma::vec bernrjct(arma::vec prb, int s) {
  int n = prb.n_elem;
  arma::vec y(n);
  arma::vec p(n);
  double t = bernroot(prb, s, -1.0, 1.0, 10);
  for (int i = 0; i < n; ++i) {
    p(i) = prb(i) * exp(t) / (prb(i) * exp(t) + 1 - prb(i));
  }
  do {
    for (int i = 0; i < n; ++i) {
      y(i) = R::rbinom(1, p(i));
    }
  } while (accu(y) != s);
  return y;
}
