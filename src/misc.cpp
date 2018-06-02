// Miscellaneous utility functions.

#include <RcppArmadillo.h>

void vswap(arma::vec & x, int a, int b) {
  double y = x(a);
  x(a) = x(b);
  x(b) = y;
}

void vswap(arma::ivec & x, int a, int b) {
  int y = x(a);
  x(a) = x(b);
  x(b) = y;
}

arma::vec repeat(arma::vec x, int n) {
  int m = x.n_elem;
  arma::vec y(m * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      y(i * m + j) = x(j);
    }
  }
  return y;
}

arma::vec repeat(arma::vec x, arma::vec n) {
  arma::vec y(accu(n));
  int m = x.n_elem;
  int t = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n(i); j++) {
      y(t) = x(i);
      t = t + 1;
    }
  }
  return y;
}

arma::vec lowertri(arma::mat x) {
  int n = x.n_rows;
  int m = x.n_cols;
  int d = std::min(n,m) * (std::min(n,m) + 1) / 2;
  if (n > m) {
    d = d + (n - m) * m;
  }
  arma::vec y(d);
  int t = 0;
  for (int j = 0; j < std::min(n,m); j++) {
    for (int i = j; i < n; i++) {
      y(t) = x(i,j);
      t = t + 1;
    }
  }
  return y;
}

arma::mat vec2symm(arma::vec x) {
  int n = (sqrt(8 * x.n_elem + 1) - 1)/2;
  int t = 0;
  arma::mat y(n, n);
  for (int j = 0; j < n; j++) {
    for (int i = j; i < n; i++) {
      y(i,j) = x(t);
      t = t + 1;
    }
  }
  return symmatl(y);
}

arma::vec invlogit(arma::vec x) {
  int n = x.n_elem;
  arma::vec p(n);
  for (int i = 0; i < n; i++) {
    p(i) = R::plogis(x(i), 0.0, 1.0, true, false);
  }
  return p;
}

arma::umat indexmat(arma::vec x) {
  if (!x.is_sorted()) Rcpp::Rcout << "warning: unsorted vector in indexmat";
  arma::vec u = unique(x);
  arma::umat y(u.n_elem, 2);
  int n = x.n_elem;
  int i = 0;
  y(i, 0) = 0;
  y(y.n_rows - 1, 1) = n - 1;
  for (int t = 1; t < n; t++) {
    if (x(t) != x(t - 1)) {
      y(i, 1) = t - 1;
      y(i + 1, 0) = t;
      i = i + 1;
    }
  }
  return y;
}
