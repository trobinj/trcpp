#include <RcppArmadillo.h>
#include "post.h"
#include "tilt.h"
#include "misc.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
List bernlong(arma::vec y, arma::mat x, arma::vec z, int m, arma::vec block, int samples, 
  arma::mat betaprior, arma::vec phivprior, double betadelt, double zetadelt) {
  
  int p = x.n_cols;
  int q = z.n_cols;
  int n = x.n_rows / m; 
  int b = max(block);
  int r = q * (q + 1) / 2;
  
  arma::uvec indx;
  arma::vec beta(p, arma::fill::zeros);
  arma::mat zeta(n, q, arma::fill::zeros);
  arma::vec zoff(size(y), arma::fill::zeros);
  arma::mat betasave(samples, p, arma::fill::zeros);
  arma::mat phivsave(samples, r, arma::fill::zeros);
  arma::vec zetatemp(q);
  
  unsigned int low, upp;
  
  arma::mat phiv(q, q);
  phiv.eye(q, q);
  
  double betamove = 0;
  double zetamove = 0;
  
  for (int i = 1; i < samples; i++) {
    
    beta = betalogrpost(x, y, zoff, beta, arma::zeros(p), betaprior, betadelt);
    betasave.row(i) = beta.t();
    if (any(betasave.row(i) != betasave.row(i - 1))) {
      betamove++;    
    }
    
    for (int j = 0; j < n; j++) {
      low = m * j;
      upp = low + m - 1;
      zetatemp = betalogrpost(z.rows(low,upp), y(arma::span(low, upp)), x.rows(low,upp) * beta, 
        zeta.row(j).t(), arma::zeros(q), phiv, zetadelt);
      if (any(zetatemp != zeta.row(j).t())) {
        zetamove++;
      }
      zoff(arma::span(low, upp)) = z.rows(low, upp) * zetatemp;
      zeta.row(j) = zetatemp.t();
    }
    
    if (q == 1) {
      phiv(0,0) = sigmpost(arma::vectorise(zeta), arma::zeros(n), phivprior(0), phivprior(1)); 
    }
    else {
      phiv = covmpost(zeta, arma::zeros(q), phivprior(0), arma::eye(q,q) * phivprior(1));
    }
    phivsave.row(i) = lowertri(phiv).t();
    
    for (int k = 1; k < (b + 1); k++) {
      indx = find(block == k);
      y(indx) = bernrjct(invlogit(x.rows(indx) * beta + zoff(indx)), sum(y(indx)));
    }
  }
  
  Rcout << "beta transition rate: " << betamove/(samples - 1) << "\n";
  Rcout << "zeta transition rate: " << zetamove/(samples - 1)/n << "\n";

  return List::create(
    Named("beta") = wrap(betasave),
    Named("phiv") = wrap(phivsave)
  );
}
  