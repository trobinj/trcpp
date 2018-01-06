// Source file for functions for sampling from a posterior distribution.

#include <RcppArmadillo.h>
#include "dist.h"

arma::vec meanpost(arma::mat y, arma::mat sigma, arma::vec mu0, arma::mat sigma0) {
  int n = y.n_rows;
  int m = y.n_cols;
  arma::mat sigma0inv(m,m);
  arma::mat sigmainv(m,n);
  arma::mat B(m,m);
  arma::vec b(m);
  sigma0inv = arma::inv(sigma0);
  sigmainv = arma::inv(sigma);
  B = arma::inv(sigma0inv + n * sigmainv);
  b = sigma0inv * mu0 + n * sigmainv * arma::mean(y).t();
  return mvrnorm(B * b, B);
}

// Note: Here alph and beta are the shape and rate parameters for the
// parameterization of the gamma distribution (i.e., 1/beta = scale). 
// Also note that the second argument of R::rgamma is scale, not rate.
double sigmpost(arma::vec y, arma::vec mu, double alph, double beta) {
  int n = y.n_elem;
  double a = alph + n / 2.0;
  double b = beta + accu(square(y - mu)) / 2.0;
  return 1 / R::rgamma(a, 1/b);
}

arma::mat covmpost(arma::mat y, arma::vec mu, int df, arma::mat scale) {
  int n = y.n_rows;
  int m = y.n_cols;
  arma::mat v(m, m, arma::fill::zeros);
  arma::vec z(m);
  for (int i = 0; i < n; i++) {
    z = y.row(i).t() - mu;
    v = v + z * z.t();
  }
  return inv(rwishart(n + df, inv(inv(scale) + v)));
}

arma::vec betapost(arma::mat x, arma::vec y, double phiv, arma::vec mb, arma::mat Rb) {
  int p = x.n_cols;
  arma::mat B(p, p);
  arma::vec b(p);
  B = inv(x.t() * x / phiv + Rb);
  b = x.t() * y / phiv + Rb * mb; 
  return mvrnorm(B * b, B);
}

arma::vec betablockpost(arma::mat x, arma::mat z, arma::vec y, arma::vec clust,
  double psiv, arma::mat Rz, arma::vec mb, arma::mat Rb) {
  
  int n = max(clust);
  int p = x.n_cols;
  int q = z.n_cols;
  
  arma::mat xwx(p, p, arma::fill::zeros);
  arma::vec xwy(p, arma::fill::zeros);
  
  arma::mat B(p, p);
  arma::vec b(p);
  
  arma::uvec indx;
  
  for (int i = 0; i < n; i++) {

    indx = find(clust == i + 1);
        
    int m = indx.n_elem;
    
    arma::mat xi(m, p);
    arma::mat zi(m, q);
    arma::vec yi(m);
    arma::mat xw(p, m);
        
    xi = x.rows(indx);
    zi = z.rows(indx);
    yi = y(indx);
    
    xw = xi.t() * inv(zi * Rz * zi.t() + arma::eye(m,m) * psiv);
    xwx = xwx + xw * xi;
    xwy = xwy + xw * yi;
  }
  
  B = inv(xwx + Rb);
  b = xwy + Rb * mb;
  
  return mvrnorm(B * b, B);
}