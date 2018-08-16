// Functions for testing purposes only. 

#include <RcppArmadillo.h>
#include <cmath>

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
void foo(arma::mat x) 
{
  int n = x.n_rows;
  int m = x.n_cols;
  Rcout << x << "\n";
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (std::isnan(x(i,j))) {
        x(i,j) = 0.0;
      } 
    }
  }
  Rcout << x << "\n";
}