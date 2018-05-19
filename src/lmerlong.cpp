#include <RcppArmadillo.h>
#include "dist.h"
#include "post.h"
#include "perm.h"
#include "misc.h"
#include "samp.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
List lmerlong(arma::mat x, arma::mat z, arma::vec y, int m, arma::vec block,
  arma::vec samples, arma::mat betaprior, arma::vec phivprior, arma::vec psivprior, 
  arma::vec vm, arma::mat vs, double delt) {
  
  int p = x.n_cols;
  int q = z.n_cols;
  int n = x.n_rows / m;    
  int b = max(block);
  int r = q * (q + 1) / 2;
  
  arma::vec clust;
  clust = repeat(arma::regspace(1, n), arma::ones(n) * m);
  
  arma::vec beta(p); 
  arma::mat zeta(n, q);
  arma::mat phiv(q, q);
  arma::vec gamm(r + 1);
  double psiv;
  
  arma::mat betasave(samples(0), p, arma::fill::zeros);
  arma::vec psivsave(samples(0), arma::fill::zeros);
  arma::mat phivsave(samples(0), q * (q + 1) / 2, arma::fill::zeros);
  
  arma::vec zvec(size(y), arma::fill::zeros);
  
  beta.fill(0.0);
  zeta.randn();
  phiv.eye(q, q);
  psiv = 1.0;
  
  arma::mat Rb = inv(betaprior);
  arma::mat Rz = inv(phiv);
  arma::vec mb(p, arma::fill::zeros);
  arma::vec mz(q, arma::fill::zeros);
  
  unsigned int low, upp;
  arma::umat bndx;
  
  for (int i = 0; i < samples(0); i++) {
    
    beta = betablockpost(x, z, y, clust, psiv, Rz, mb, Rb);
    betasave.row(i) = beta.t();

    if (delt > 0) {
      gamm.head(r) = lowertri(phiv);
      gamm(r) = psiv;
      gamm = varmsamp(y, x, z, m, beta, mb, Rb, gamm, vm, vs, phivprior, psivprior, delt);
      phiv = vec2symm(gamm.head(r));
      psiv = gamm(r);
    }
    else {
      for (int j = 0; j < n; j++) {
        low = m * j;
        upp = low + m - 1;
        zeta.row(j) = betapost(z.rows(low, upp), y(arma::span(low, upp)) - x.rows(low, upp) * beta, psiv, mz, Rz).t();
        zvec(arma::span(low, upp)) = z.rows(low, upp) * zeta.row(j).t();
      }
      psiv = sigmpost(y, x * beta + zvec, psivprior(0), psivprior(1));
      if (q == 1) {
        phiv(0,0) = sigmpost(arma::vectorise(zeta), arma::zeros(n), phivprior(0), phivprior(1)); 
      }
      else {
        phiv = covmpost(zeta, arma::zeros(q), phivprior(0), arma::eye(q,q) * phivprior(1));
      }
    }
    psivsave(i) = psiv;
    phivsave.row(i) = lowertri(phiv).t(); 
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

