// Functions for tilted rejection sampling.

#include <RcppArmadillo.h>

double bernscor(arma::vec prb, double t, int s) {
  int n = prb.n_elem;
  double y = 0.0;
  for (int i = 0; i < n; i++) {
    y = y + prb(i) * exp(t) / (prb(i) * exp(t) + 1 - prb(i));
  }
  return s - y;
}

double bernroot(arma::vec prb, int s, double a, double b, double n) {
  double fa, fb, c = 0.0;
  for (int i = 0; i < n; i++) {
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
  for (int i = 0; i < n; i++) {
    p(i) = prb(i) * exp(t) / (prb(i) * exp(t) + 1 - prb(i));
  }
  do {
    for (int i = 0; i < n; i++) {
      y(i) = R::rbinom(1, p(i));
    }
  } while (accu(y) != s);
  return y;
}