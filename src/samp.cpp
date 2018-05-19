// Miscellaneous samplers.

#include <RcppArmadillo.h>
#include "dist.h"
#include "misc.h"
#include "logl.h"

const double log2pi = log(2.0 * M_PI);

using namespace Rcpp;

double tnsigma(arma::vec y, double sold, double mu, double sigma, double a, double b) {
 
  double snew;
  snew = rtnorm(mu, sigma, a, b);

  double numlogdens, denlogdens;
  numlogdens = normlogl(y, 0.0, snew);
  denlogdens = normlogl(y, 0.0, sold);

  if (R::runif(0.0, 1.0) < exp(numlogdens - denlogdens)) {
    return snew;
  }
  else {
    return sold;
  }
}

double lmerlogl(arma::vec y, arma::mat x, arma::mat z, int m, 
  arma::vec beta, arma::mat phiv, double psiv) {
  
  double logl = 0.0, lds, sign;
  int n = y.n_elem/m;
  int p = x.n_cols;
  int q = z.n_cols;
  arma::vec yi(m);
  arma::mat xi(m,p);
  arma::mat zi(m,q);
  arma::mat si(m,m);
  for (int i = 0; i < n; i++) {
    yi = y(arma::span(m * i, m * i + m - 1));
    xi = x.rows(m * i, m * i + m - 1);
    zi = z.rows(m * i, m * i + m - 1);
    si = zi * phiv * zi.t() + psiv * arma::eye(m,m);
    log_det(lds, sign, si);
    logl = logl - (lds + as_scalar((yi - xi*beta).t() * inv(si) * (yi - xi*beta))) / 2.0;
  }
  return logl;
}

bool posdef(arma::mat phi, double psi) {
  arma::vec eigval;
  arma::mat eigvec;
  eig_sym(eigval, eigvec, phi);
  return all(eigval > 0) * (psi > 0);
}

arma::vec varmsamp(arma::vec y, arma::mat x, arma::mat z, double m,  
  arma::vec beta, arma::vec mb, arma::mat Rb, arma::vec vold, 
  arma::vec vm, arma::mat vs, arma::vec phivprior, arma::vec psivprior, double delt) {

  int q = z.n_cols;
  int r = q * (q + 1) / 2;

  double psiv;
  arma::mat phiv(q, q);

  double loglold;  
  phiv = vec2symm(vold.head(r));
  psiv = vold(r);

  loglold = lmerlogl(y, x, z, m, beta, phiv, psiv);
  if (q == 1) {
    loglold = loglold + R::dgamma(1/as_scalar(phiv), phivprior(0), 1/phivprior(1), true)
      + R::dgamma(1/psiv, psivprior(0), 1/psivprior(1), true);
  }
  else {
    loglold = loglold + dwishart(inv(phiv), phivprior(0), arma::eye(size(phiv)) * phivprior(1), true)
      + R::dgamma(1/psiv, psivprior(0), 1/psivprior(1), true);
  }

  double loglnew;
  arma::vec vnew(size(vold));
  do {
    vnew = rmvt(vm, vs, delt);  
    phiv = vec2symm(vnew.head(r));
    psiv = vnew(r);
  } while (!posdef(phiv, psiv));
  loglnew = lmerlogl(y, x, z, m, beta, phiv, psiv);
  if (q == 1) {
    loglnew = loglnew + R::dgamma(1/as_scalar(phiv), phivprior(0), 1/phivprior(1), true)
      + R::dgamma(1/psiv, psivprior(0), 1/psivprior(1), true);
  }
  else {
    loglnew = loglnew + dwishart(inv(phiv), phivprior(0), arma::eye(size(phiv)) * phivprior(1), true)
      + R::dgamma(1/psiv, psivprior(0), 1/psivprior(1), true);
  }

  double logprob;
  logprob = loglnew + dmvt(vold, vm, vs, delt, true) - loglold - dmvt(vnew, vm, vs, delt, true);

  if (R::runif(0.0, 1.0) < exp(logprob)) {
    return vnew; 
  }
  else {
    return vold;
  }
}
 
