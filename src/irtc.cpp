// Functions for block-censored item response models.

#include <RcppArmadillo.h>
#include "logl.h"
#include "misc.h"
#include "post.h"

using namespace Rcpp;

// Censoring rule.
bool censor(arma::mat y, double b) {
  return sum(y,1).min() < b ? true : false; 
}

// Sample block of item responses from proposal distribution. 
arma::mat impute1pl(arma::vec zeta, arma::vec delt) {
  int n = zeta.n_elem;
  int m = delt.n_elem;
  arma::mat y(n,m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      y(i,j) = R::rbinom(1, invlogit(zeta[i] + delt[j]));
    }
  }
  return y;
}

// Metropolis-Hastings sampling step for a respondent or item parameter for a one-parameter logistic model. 
double samp1pl(arma::vec y, double pold, arma::vec offset, double pm, double ps, double tune, int &cnt) {
  double pnew = R::rnorm(pold, tune);
  double lold = bernlogl(y, invlogit(pold + offset)) + R::dnorm(pold, pm, ps, true);
  double lnew = bernlogl(y, invlogit(pnew + offset)) + R::dnorm(pnew, pm, ps, true);
  if (R::runif(0.0, 1.0) < exp(lnew - lold)) {
    ++cnt;
    return pnew;
  } else {
    return pold;
  }
}

//' @export
// [[Rcpp::export]]
List mcmc1pl(arma::mat y, arma::mat x, arma::vec d, int samp, double dtune, double ztune) {

  int n = y.n_rows;
  int m = y.n_cols;
  int p = x.n_cols;
  int b = max(d);
  
  int dcnt = 0;
  int zcnt = 0;

  arma::vec zeta(n, arma::fill::zeros);
  arma::vec delt(m, arma::fill::zeros);
  arma::vec beta(p, arma::fill::zeros);
  double psiv = 1.0;

  arma::mat deltsave(samp, m);
  arma::mat betasave(samp, p);
  arma::vec psivsave(samp);
  
  for (int t = 0; t < samp; ++t) {
    
    if ((t + 1) % 1000 == 0) {
      Rcout << "Sample: " << t + 1 << "\n";
    }
    
    // Sample respondent parameters.
    for (int i = 0; i < n; ++i) {
      zeta(i) = samp1pl(vectorise(y.row(i)), zeta(i), delt, as_scalar(x.row(i) * beta), sqrt(psiv), ztune, zcnt);
    } 
    
    // Sample item parameters.
    for (int j = 0; j < m; ++j) {
      delt(j) = samp1pl(y.col(j), delt(j), zeta, 0.0, 3.0, dtune, dcnt); // prior specified here
      deltsave(t,j) = delt(j);
    }

    // Sample covariate parameters.
    beta = betapost(x, zeta, psiv, arma::zeros(p), arma::eye(p,p) * sqrt(3)); // prior specified here
    betasave.row(t) = beta.t();

    // Sample respondent variable variance (needs to be modified when using covariates).
    psiv = sigmpost(zeta, x * beta, 10.0/2, 10.0/2); // prior specified here
    psivsave(t) = psiv;
        
    // Sample censored item responses using a rejection sampler.
    for (int k = 1; k < (b + 1); ++k) {
      arma::uvec indx = find(d == k);
      arma::mat ytmp = y.rows(indx);
      arma::vec ztmp = zeta(indx);
      do {
        ytmp = impute1pl(ztmp, delt);
      } while (!censor(ytmp, 2.5));
      y.rows(indx) = ytmp;
    }
  }

  Rcout << "zeta acceptance rate: " << double(zcnt) / (samp * n) << "\n";
  Rcout << "delt acceptance rate: " << double(dcnt) / (samp * m) << "\n";

  return List::create(
    Named("delt") = wrap(deltsave),
    Named("beta") = wrap(betasave),
    Named("psiv") = wrap(psivsave)
  );
}
