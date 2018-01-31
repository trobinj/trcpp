// Miscellaneous samplers.

#include <RcppArmadillo.h>
#include "dist.h"
#include "misc.h"

const double log2pi = log(2.0 * M_PI);

using namespace Rcpp;

double varmlogl(arma::vec y, arma::mat x, arma::mat z, double m, 
  arma::mat phiv, double psiv, arma::vec beta, arma::vec mb, arma::mat Rb) {

  int n = y.n_elem / m;
  int p = x.n_cols;
  int q = z.n_cols;
  
  arma::vec yi(m);
  arma::mat xi(m, p);
  arma::vec zi(m, q);
  arma::mat si(m, m);
  
  arma::mat xw(p, m);
  arma::mat xwx(p, p, arma::fill::zeros);
  arma::vec xwy(p, arma::fill::zeros);

  arma::vec b(p);
  arma::mat B(p, p); 
  
  arma::mat Rz(q, q);
  Rz = inv(phiv);
  
  double logl = 0.0, lds, sign;    
  for (int i = 0; i < n; i++) {
    yi = y(arma::span(m * i, m * i + m - 1));
    xi = x.rows(m * i, m * i + m - 1);
    zi = z.rows(m * i, m * i + m - 1);
    
    si = zi * phiv * zi.t() + psiv * arma::eye(m,m);
    log_det(lds, sign, si);
    logl = logl - (log2pi + lds + as_scalar((yi - xi*beta).t() * inv(si) * (yi - xi*beta)))/2;
    
    xw = xi.t() * inv(zi * Rz * zi.t() + arma::eye(m, m) * psiv);
    xwx = xwx + xw * xi;
    xwy = xwy + xw * yi;
  }
  
  B = inv(xwx/psiv + Rb);
  b = xwy/psiv + Rb * mb;
  
  double t1, t2, t3;
  t1 = dmvnorm(beta, mb, inv(Rb), true);
  t2 = logl;
  t3 = dmvnorm(beta, B*b, B, true);
  
  return t1 + t2 - t3;
}

bool posdef(arma::mat phi, double psi) {
  arma::vec eigval;
  arma::mat eigvec;
  eig_sym(eigval, eigvec, phi);
  return all(eigval > 0) * (psi > 0);
}

arma::vec varmsamp(arma::vec y, arma::mat x, arma::mat z, double m, 
  arma::vec beta, arma::vec mb, arma::mat Rb, 
  arma::vec vold, arma::vec vm, arma::mat vs,
  arma::vec phivprior, arma::vec psivprior) {

  int q = z.n_cols;
  int r = q * (q + 1) / 2;
  
  double psiv;
  arma::mat phiv(q, q);
  
  double loglold;  
  phiv = vec2symm(vold.head(r));
  psiv = vold(r);
  
  loglold = varmlogl(y, x, z, m, phiv, psiv, beta, mb, Rb);
  if (q == 1) {
    loglold = loglold + R::dgamma(1/as_scalar(phiv), phivprior(0), phivprior(1), true)
      + R::dgamma(1/psiv, psivprior(0), psivprior(1), true);
  }
  else {
    loglold = loglold + dwishart(inv(phiv), phivprior(0), arma::eye(size(phiv)) * phivprior(1), true)
      + R::dgamma(1/psiv, psivprior(0), psivprior(1), true);
  }
  
  double loglnew;
  arma::vec vnew(size(vold));
  do {
    vnew = mvrnorm(vm, vs);  
    phiv = vec2symm(vnew.head(r));
    psiv = vnew(r);
  } while (!posdef(phiv, psiv));
  loglnew = varmlogl(y, x, z, m, phiv, psiv, beta, mb, Rb);
  if (q == 1) {
    loglnew = loglnew + R::dgamma(1/as_scalar(phiv), phivprior(0), phivprior(1), true)
    + R::dgamma(1/psiv, psivprior(0), psivprior(1), true);
  }
  else {
    loglnew = loglnew + dwishart(inv(phiv), phivprior(0), arma::eye(size(phiv)) * phivprior(1), true)
    + R::dgamma(1/psiv, psivprior(0), psivprior(1), true);
  }
  
  double logprob;
  logprob = loglnew + dmvnorm(vold, vm, vs, true) - loglold - dmvnorm(vnew, vm, vs, true);
  
  if (R::runif(0.0, 1.0) < exp(logprob)) {
    return vnew; 
  }
  else {
    return vold;
  }
}

/*
arma::vec phipsi(arma::vec y, arma::mat z, arma::umat indx, double phi, double psi,
  arma::vec mprop, arma::mat sprop, arma::vec phivprior, arma::vec psivprior) {
  arma::vec vnew(2);
  vnew = exp(mvrnorm(mprop, sprop));
  
  arma::vec vold(2);
  vold(0) = phi;
  vold(1) = psi;
  
  int low, upp, m;
  int n = indx.n_rows;
  int q = z.n_cols;
  
  double loglold = 0.0;
  double loglnew = 0.0;
  
  for (int i = 0; i < n; i++) {
    
    low = indx(i, 0);
    upp = indx(i, 1);
    m = upp - low + 1;
    
    arma::mat zi(m, q);
    arma::vec yi(m);
    arma::mat cvarold(m, m);
    arma::mat cvarnew(m, m);

    zi = z.rows(low, upp);
    yi = y(arma::span(low, upp));
        
    cvarold = zi * (arma::ones(1,1) * vold(0)) * zi.t() + arma::eye(m,m) * vold(1);
    cvarnew = zi * (arma::ones(1,1) * vnew(0)) * zi.t() + arma::eye(m,m) * vnew(1);
    
    loglold = loglold + dmvnorm(yi, arma::zeros(m), cvarold, true);
    loglnew = loglnew + dmvnorm(yi, arma::zeros(m), cvarnew, true);
  }

  loglold = loglold 
    + R::dgamma(1/vold(0), phivprior(0), phivprior(1), true) 
    + R::dgamma(1/vold(1), psivprior(0), psivprior(1), true);
  loglnew = loglnew 
    + R::dgamma(1/vnew(0), phivprior(0), phivprior(1), true) 
    + R::dgamma(1/vnew(1), psivprior(0), psivprior(1), true);

  
  if (R::runif(0.0, 1.0) < exp(loglnew - loglold)) {
    return vnew;
  }
  else {
    return vold;
  }
  
}
 
 */