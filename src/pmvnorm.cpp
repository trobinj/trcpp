#include <RcppArmadillo.h>

using namespace Rcpp;

double pnorm(double z) {
  return R::pnorm(z, 0.0, 1.0, true, false);
}

double qnorm(double p) {
  return R::qnorm(p, 0.0, 1.0, true, false);
}

// Note: This function is currenty broken. Do not use. 
//' @export
// [[Rcpp::export]]
double pmvnorm(arma::mat s, arma::vec a, arma::vec b, double epsi, double alph, int nmax) {
  arma::mat c = chol(s, "lower");
  double intsum = 0.0, varsum = 0.0, delt, cy;
  int m = a.n_elem;
  arma::vec y(m - 1);
  arma::vec w(m - 1);
  arma::vec d(m - 1); d(0) = pnorm(a(0) / c(1,1));
  arma::vec e(m - 1); e(0) = pnorm(b(0) / c(1,1));
  arma::vec f(m - 1); f(0) = e(0) - d(0);
  for (int i = 0; i < nmax; ++i) {
    w.randu();
    for (int j = 1; j < m; ++j) {
      y(j - 1) = qnorm(d(j - 1) + w(j - 1) * (e(j - 1) - d(j - 1)));
      cy = 0.0;
      for (int k = 0; k < j; ++k) {
        cy = c(j, k) * y(k);
      }
      d(j) = pnorm((a(j) - cy) / c(j, j));
      e(j) = pnorm((b(j) - cy) / c(j, j));
      f(j) = (e(j) - d(j)) * f(j - 1);
    }
    delt = (f(m - 1) - intsum)/(i + 1);
    intsum = intsum + delt;
    varsum = (i + 1 - 2) * varsum / (i + 1) + pow(delt, 2);
    if (alph * sqrt(varsum) < epsi) {
      break;
    }
  }
  return intsum;
}