// Miscellaneous utility functions.

#include <RcppArmadillo.h>
#include <string>

void mssg(std::string x) {
  Rcpp::Rcout << x << "\n";
}

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

void fill(arma::mat & y, arma::vec x) {
  int n = y.n_rows;
  int m = y.n_cols;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      y(i,j) = x(y(i,j));
    }
  }
}

arma::mat expand(arma::uvec x) {
  int n = prod(x);
  int m = x.n_elem;
  int z, s, t;
  arma::mat y(n,m);
  for (int j = 0; j < m; ++j) {
    z = prod(x.subvec(0,j));
    s = 1;
    t = 1;
    for (int i = 0; i < n; ++i) {
      if (s + 1 > n / z) {
        s = 1;
        y(i,j) = t;
        if (t + 1 > x(j)) {
          t = 1;
        } else {
          ++t;
        }
      } else {
        y(i,j) = t;
        ++s;
      }
    }
  }
  return y;
}

arma::uvec rankvec(arma::vec x) {
  int n = x.n_elem;
  arma::uvec q = sort_index(x);
  arma::uvec r(n);
  for (unsigned int i = 0; i < n; ++i) {
    r(q(i)) = i + 1;
  }
  return r;
}

arma::vec repeat(arma::vec x, int n) {
  int m = x.n_elem;
  arma::vec y(m * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      y(i * m + j) = x(j);
    }
  }
  return y;
}

arma::vec repeat(arma::vec x, arma::vec n) {
  arma::vec y(accu(n));
  int m = x.n_elem;
  int t = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n(i); ++j) {
      y(t) = x(i);
      ++t;
    }
  }
  return y;
}

arma::vec lowertri(arma::mat x, bool diag) {
  if (!diag) x.shed_row(0);
  int n = x.n_rows;
  int m = x.n_cols;
  int d = std::min(n,m) * (std::min(n,m) + 1) / 2;
  if (n > m) {
    d = d + (n - m) * m;
  }
  arma::vec y(d);
  int t = 0;
  for (int j = 0; j < std::min(n,m); ++j) {
    for (int i = j; i < n; ++i) {
      y(t) = x(i,j);
      ++t;
    }
  }
  return y;
}

arma::mat vec2symm(arma::vec x) {
  int n = (sqrt(8 * x.n_elem + 1) - 1)/2;
  int t = 0;
  arma::mat y(n, n);
  for (int j = 0; j < n; ++j) {
    for (int i = j; i < n; ++i) {
      y(i,j) = x(t);
      ++t;
    }
  }
  return symmatl(y);
}

double invlogit(double x) {
  return R::plogis(x, 0.0, 1.0, true, false);
}

arma::vec invlogit(arma::vec x) {
  int n = x.n_elem;
  arma::vec p(n);
  for (int i = 0; i < n; ++i) {
    p(i) = R::plogis(x(i), 0.0, 1.0, true, false);
  }
  return p;
}

arma::umat indexmat(arma::vec x) {
  if (!x.is_sorted()) Rcpp::Rcout << "warning: unsorted vector" << "\n";
  arma::vec u = unique(x);
  arma::umat y(u.n_elem, 2);
  int n = x.n_elem;
  int i = 0;
  y(i, 0) = 0;
  y(y.n_rows - 1, 1) = n - 1;
  for (int t = 1; t < n; ++t) {
    if (x(t) != x(t - 1)) {
      y(i, 1) = t - 1;
      y(i + 1, 0) = t;
      ++i;
    }
  }
  return y;
}

// Compute nodes and weights for the Gauss-Hermite quadrature approximation
// \int_{-\infty}^{\infty} e^{-x^2} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)
// using the Golub-Welsch algorithm.
void ghquad(int n, arma::vec & node, arma::vec & wght) {
  arma::mat J(n, n, arma::fill::zeros);
  for (int i = 0; i < (n - 1); ++i) {
    J(i + 1, i) = sqrt((i + 1) / 2.0);
    J(i, i + 1) = J(i + 1, i);
  }
  const double b0 = sqrt(M_PI);
  arma::vec eigenval(n);
  arma::mat eigenvec(n, n);
  arma::eig_sym(eigenval, eigenvec, J);
  for (int i = 0; i < n; ++i) {
    node(i) = eigenval(i);
    wght(i) = b0 * pow(eigenvec(0, i), 2) / pow(arma::norm(eigenvec.col(i)), 2);
  }
}
/*
 * To do: Extend the above for other kinds of Gaussian quadrature.
 */
