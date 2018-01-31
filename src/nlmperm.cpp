#include <RcppArmadillo.h>
#include "post.h"
#include "perm.h"
#include "misc.h"

using namespace Rcpp;

List nlmperm(arma::mat x, arma::vec y, arma::uvec block, 
  arma::vec samples, arma::mat betaprior, arma::vec psiprior) {
  
  int p = x.n_cols;
  int b = max(block);
  
  arma::mat betasave(samples(0), p);
  arma::vec psivsave(samples(0));
  arma::vec beta(p);
  arma::uvec indx;
  
  double psiv = 1.0;

  betasave.fill(0.0);
  psivsave.fill(1.0);
  
  arma::mat Rb(p, p);
  arma::vec mb(p);
  Rb = inv(betaprior);
  mb.fill(0.0);
  
  for (int i = 1; i < samples(0); i++) {
    
    beta = betapost(x, y, psiv, mb, Rb);
    betasave.row(i) = beta.t();
    
    psiv = sigmpost(y, x * beta, psiprior(0), psiprior(1));
    psivsave(i) = psiv;
    
    for (int k = 1; k < (b + 1); k++) {
      indx = find(block == k);
      y(indx) = normperm(samples(1), y(indx), x.rows(indx) * beta, sqrt(psiv));
    }
  }
  
  return List::create(
    Named("beta") = wrap(betasave),
    Named("psiv") = wrap(psivsave)
  );
}