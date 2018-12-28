// Miscellaneous functions for solving combinatoric problems.

#include <RcppArmadillo.h>
#include "misc.h"

arma::vec pairperm(arma::vec y) {
  int n = y.n_elem;

  arma::uvec x(n);
  x.fill(2);

  arma::mat d = expand(x);
  int m = d.n_rows;

  arma::vec z(2);
  z(0) = -1;
  z(1) =  1;
  d = d - 1;
  fill(d, z);

  arma::vec s(m);

  for (int i = 0; i < m; ++i) {
    s(i) = as_scalar(d.row(i) * y) / static_cast<double>(n);
  }
  return s;
}
