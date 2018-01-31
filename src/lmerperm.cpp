#include <RcppArmadillo.h>
#include "post.h"
#include "perm.h"
#include "misc.h"
#include "samp.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
List lmerperm(arma::mat x, arma::mat z, arma::vec y, arma::vec clust, arma::vec block,
  arma::vec samples, arma::mat betaprior, arma::vec phivprior, arma::vec psivprior) {
  
  int p = x.n_cols;
  int q = z.n_cols;
  int n = max(clust);    
  int b = max(block);
  
  arma::mat betasave(samples(0), p, arma::fill::zeros);
  arma::vec psivsave(samples(0), arma::fill::zeros);
  arma::mat phivsave(samples(0), q * (q + 1) / 2, arma::fill::zeros);
  
  arma::vec beta(p); 
  arma::mat zeta(n, q);
  arma::mat phiv(q, q);
  double psiv;
  
  arma::vec zvec(size(y), arma::fill::zeros);
  
  beta.fill(0.0);
  zeta.fill(0.0);
  phiv.eye(q, q);
  psiv = 1.0;
  
  arma::mat Rb = inv(betaprior);
  arma::mat Rz = inv(phiv);
  arma::vec mb(p, arma::fill::zeros);
  arma::vec mz(q, arma::fill::zeros);

  arma::umat indx = indexmat(clust);
  unsigned int low, upp;
  arma::umat bndx;
    
  arma::vec temp(2);
  arma::vec mpri(2);
  arma::mat spri(2,2);
    
  for (int i = 0; i < samples(0); i++) {
    
    beta = betablockpost(x, z, y, clust, psiv, Rz, mb, Rb);
    betasave.row(i) = beta.t();
    
    for (int j = 0; j < n; j++) {
      low = indx(j, 0);
      upp = indx(j, 1);
      zeta.row(j) = betapost(z.rows(low, upp), y(arma::span(low, upp)) - x.rows(low, upp) * beta, psiv, mz, Rz).t();
      zvec(arma::span(low, upp)) = z.rows(low, upp) * zeta.row(j).t();
    }
    
    psiv = sigmpost(y, x * beta + zvec, psivprior(0), psivprior(1));
    psivsave(i) = psiv;
      
    if (q == 1) {
      phiv(0,0) = sigmpost(arma::vectorise(zeta), arma::zeros(n), phivprior(0), phivprior(1)); 
    }
    else {
      phiv = covmpost(zeta, arma::zeros(q), phivprior(0), arma::eye(q,q) * phivprior(1));
    }
    phivsave.row(i) = lowertri(phiv).t(); // is this correct?
    Rz = inv(phiv);

    for (int k = 1; k < (b + 1); k++) {
      bndx = find(block == k);
      y(bndx) = normperm(samples(1), y(bndx), x.rows(bndx) * beta + zvec(bndx), sqrt(psiv));
    }
  }
  
  return List::create(
    Named("beta") = wrap(betasave),
    Named("psiv") = wrap(psivsave),
    Named("phiv") = wrap(phivsave)
  );
}

