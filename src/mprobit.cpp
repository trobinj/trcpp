// MCMC algorithm for a multivariate probit model with unconstrained correlation structure based on an
// algorithm proposed by Talhouk et al. (2012, Journal of Computational and Graphical Statistics), using
// a marginally uniform prior on the correlation matrix proposed by Barnard et al. (2000, Statistica Sinica).

#include <RcppArmadillo.h>
#include "dist.h" // for mvrnorm, rtnormpos, and rwishart
#include "misc.h" // for lowertri

using namespace Rcpp;

// Function to compute conditional variance.
arma::vec cdists(arma::mat s) {
  int n = s.n_cols;
  arma::vec y(n);
  for (int j = 0; j < n; j++) {
    arma::mat s22 = s;
    s22.shed_row(j);
    s22.shed_col(j);
    s22 = inv(s22);
    arma::mat s12 = s.row(j);
    s12.shed_col(j);
    y(j) = s(j,j) - as_scalar(s12 * s22 * s12.t());
  }
  return y;
}

// Function to compute conditional mean.
double cdistm(arma::vec m, arma::mat s, arma::vec x, int j) {
  int n = s.n_cols;
  arma::mat s22 = s;
  s22.shed_row(j);
  s22.shed_col(j);
  s22 = inv(s22);
  arma::mat s12 = s.row(j);
  s12.shed_col(j);
  arma::mat m2(n, 1);
  m2.col(0) = m;
  m2.shed_row(j);
  arma::mat x2(n, 1);
  x2.col(0) = x;
  x2.shed_row(j);
  return m(j) + as_scalar(s12 * s22 * (x2 - m2)); 
}

//' @export
// [[Rcpp::export]]
List mprobit(arma::mat Y, arma::mat X, int samples) {
  int n = Y.n_rows;
  int m = Y.n_cols;
  int p = X.n_cols;
  
  arma::mat M(n, m);
  arma::mat B(p, m, arma::fill::zeros);
  arma::mat Z(n, m, arma::fill::zeros);
  arma::mat W(n, m, arma::fill::zeros);
  arma::mat D(m, m, arma::fill::eye);
  arma::mat R(m, m, arma::fill::eye);
  arma::mat S(m, m, arma::fill::eye);
  arma::mat G(p, m, arma::fill::zeros);
  arma::mat U(n, m);
  
  arma::vec r(m);
  
  arma::mat T(p, p);
  T = inv(X.t() * X + inv(arma::eye(p, p))); // note prior specification here
    
  arma::mat Bsave(samples, p * m, arma::fill::zeros);
  arma::mat Rsave(samples, m * (m + 1) / 2);
  
  double mij;
  arma::vec sij;
  
  for (int k = 0; k < samples; k++) {
    
    // Sample latent responses.
    
    M = X * B;
    sij = sqrt(cdists(R));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        mij = cdistm(vectorise(M.row(i)), R, vectorise(Z.row(i)), j); 
        if (Y(i, j) < 0) {
          Z(i, j) = R::rnorm(mij, sij(j)); // missing data coded as any negative number
        } else {
          Z(i, j) = rtnormpos(mij, sij(j), Y(i, j) == 1);
        }
      }
    }

    // Sample variances and covariances.
    
    r = diagvec(inv(R));
    for (int j = 0; j < m; j++) {
      D(j, j) = 1 / sqrt(R::rgamma((m + 1) / 2.0, r(j) / 2.0));
    }
    W = Z * D;    
    U = T * X.t() * W;
    S = inv(rwishart(n + 2, inv(W.t() * W + arma::eye(m,m) - U.t() * inv(T) * U)));
    
    // Sample beta parameters.
    
    G = mvrnorm(U, T, D * R * D);
    
    // Standardize beta parameters and covariances into correlations. 
    
    D = inv(sqrt(diagmat(S)));
    B = G * D;
    R = D * S * D;
    
    // Save sampled parameters.
    
    Bsave.row(k) = vectorise(B).t();
    Rsave.row(k) = lowertri(R).t();
  }
  
  return List::create(
    Named("beta") = wrap(Bsave),
    Named("corr") = wrap(Rsave)
  );
}