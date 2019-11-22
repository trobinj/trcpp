#include <RcppArmadillo.h>
#include "post.h"
#include "perm.h"
#include "misc.h"

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
List lmerlong(arma::vec y, arma::mat x, arma::mat z, int m, arma::vec block,
  arma::vec samples, arma::mat betaprior, arma::vec phivprior, arma::vec psivprior) {

  int p = x.n_cols;
  int q = z.n_cols;
  int n = x.n_rows / m;
  int b = max(block);

  arma::vec clust;
  clust = repeat(arma::regspace(1, n), arma::ones(n) * m);

  arma::vec beta(p, arma::fill::zeros);
  arma::mat zeta(n, q, arma::fill::zeros);
  arma::mat phiv(q, q, arma::fill::eye);
  double psiv = 1.0;

  arma::mat betasave(samples(0), p, arma::fill::zeros);
  arma::vec psivsave(samples(0), arma::fill::zeros);
  arma::mat phivsave(samples(0), q * (q + 1) / 2, arma::fill::zeros);

  arma::vec zoff(size(y), arma::fill::zeros);

  arma::mat Rb = inv(betaprior);
  arma::mat Rz = inv(phiv);
  arma::vec mb(p, arma::fill::zeros);
  arma::vec mz(q, arma::fill::zeros);

  unsigned int low, upp;
  arma::umat indx;

  if ((phivprior(0) <= 0) || (phivprior(1) <= 0)) {
    phiv.fill(0.0);
  }

  for (int i = 0; i < samples(0); ++i) {

    beta = betablockpost(x, z, y, clust, psiv, phiv, mb, Rb);
    betasave.row(i) = beta.t();

    if ((phivprior(0) > 0) && (phivprior(1) > 0)) {
      for (int j = 0; j < n; ++j) {
        low = m * j;
        upp = low + m - 1;
        zeta.row(j) = betapost(z.rows(low, upp), y(arma::span(low, upp)) - x.rows(low, upp) * beta, psiv, mz, Rz).t();
        zoff(arma::span(low, upp)) = z.rows(low, upp) * zeta.row(j).t();
      }
    }

    psiv = sigmpost(y, x * beta + zoff, psivprior(0), psivprior(1));
    psivsave(i) = psiv;
    if ((phivprior(0) > 0) && (phivprior(1) > 0)) {
      if (q == 1) {
        phiv(0,0) = sigmpost(arma::vectorise(zeta), arma::zeros(n), phivprior(0), phivprior(1));
      }
      else {
        phiv = covmpost(zeta, arma::zeros(q), phivprior(0), arma::eye(q,q) * phivprior(1));
      }
      phivsave.row(i) = lowertri(phiv).t();
      Rz = inv(phiv);
    }

    for (int k = 1; k < (b + 1); ++k) {
      indx = find(block == k);
      y(indx) = normperm(samples(1), y(indx), x.rows(indx) * beta + zoff(indx), sqrt(psiv));
    }
  }

  return List::create(
    Named("beta") = wrap(betasave),
    Named("psiv") = wrap(psivsave),
    Named("phiv") = wrap(phivsave)
  );
}
