#include <RcppArmadillo.h>
#include "post.h"
#include "perm.h"
#include "misc.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
List nlmmsamp(arma::mat x, arma::mat z, arma::vec y, arma::vec clust, arma::vec block,
  arma::vec samples, arma::mat betaprior, arma::vec phivprior, arma::vec psivprior) {
  
  int p = x.n_cols;
  int q = z.n_cols;
  int n = max(clust);    
  int b = max(block);
  int m = x.n_rows/n;    
  
  arma::uvec indx;
  
  arma::mat betasave(samples(0), p, arma::fill::zeros);
  arma::vec psivsave(samples(0), arma::fill::zeros);
  arma::mat phivsave(samples(0), q * (q + 1) / 2, arma::fill::zeros);
  
  arma::mat zeta(n, q);
  arma::vec beta(p); 
  arma::mat phiv(q, q);
  double psiv;
  
  arma::vec zvec(n*m, arma::fill::zeros);
  
  beta.fill(0.0);
  zeta.fill(0.0);
  phiv.eye(q, q);
  psiv = 1.0;
  
  arma::mat Rb = inv(betaprior);
  
  for (int i = 0; i < samples(0); i++) {
    
    for (int j = 0; j < n; j++) {
      indx = find(clust == j);
      zeta.row(j) = betapost(z.rows(indx), y(indx) - x.rows(indx) * beta, psiv, arma::zeros(q), inv(phiv)).t();
      zvec(indx) = z.rows(indx) * zeta.row(j).t();
    }
    
    beta = betablockpost(x, z, y, arma::eye(m,m) * psiv, inv(phiv), arma::zeros(p), Rb);
    betasave.row(i) = beta.t();
    
    psiv = sigmpost(y, x * beta + zvec, psivprior(0), psivprior(1));
    psivsave(i) = psiv;
    
    if (q == 1) {
      phiv(0,0) = sigmpost(arma::vectorise(zeta), arma::zeros(n), phivprior(0), phivprior(1)); 
    }
    else {
      phiv = covmpost(zeta, arma::zeros(q), phivprior(0), arma::eye(q,q) * phivprior(1));
    }
    phivsave.row(i) = lowertri(phiv).t(); // is this correct?
    
    for (int k = 1; k < (b + 1); k++) {
      indx = find(block == k);
      y(indx) = normperm(samples(1), y(indx), x.rows(indx) * beta + zvec(indx), sqrt(psiv));
    }
  }
  
  return List::create(
    Named("beta") = wrap(betasave),
    Named("psiv") = wrap(psivsave),
    Named("phiv") = wrap(phivsave)
  );
}