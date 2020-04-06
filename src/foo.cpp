// Functions for testing purposes only.

#include <RcppArmadillo.h>

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
void foo(arma::vec x, double a, double b) {
  Rcpp::Rcout << x(find(x > a && x < b));
}
