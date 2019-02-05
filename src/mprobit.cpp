// MCMC algorithm for a multivariate probit model with unconstrained correlation
// structure based on an algorithm proposed by Talhouk et al. (2012, Journal of
// Computational and Graphical Statistics), using a marginally uniform prior on
// the correlation matrix proposed by Barnard et al. (2000, Statistica Sinica).

#include <RcppArmadillo.h>
#include "dist.h" // for mvrnorm, rnormpos, and rwishart
#include "misc.h" // for lowertri

using namespace Rcpp;

// Function to compute conditional "regression coefficients".
arma::mat cdistb(arma::mat s) {
  int n = s.n_cols;
  arma::mat y(n, n - 1);
  for (int j = 0; j < n; ++j) {
    arma::mat s22 = s;
    s22.shed_row(j);
    s22.shed_col(j);
    arma::mat s12 = s.row(j);
    s12.shed_col(j);
    y.row(j) = s12 * inv(s22);
  }
  return y;
}

// Function to compute conditional mean.
double cdistm(arma::vec m, arma::mat b, arma::vec x, int j) {
  arma::mat m2(m); m2.shed_row(j);
  arma::mat x2(x); x2.shed_row(j);
  return m(j) + as_scalar(b.row(j) * (x2 - m2));
}

// Function to compute conditional variance.
arma::vec cdists(arma::mat s) {
  int n = s.n_cols;
  arma::vec y(n);
  for (int j = 0; j < n; ++j) {
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

bool reject(arma::vec y, int d, int miny, int maxy) {
  int sum = int(accu(y));
  if (std::min(sum, maxy) != d && std::max(sum, miny) != d) {
    return true;
  }
  return false;
 }

//' @export
// [[Rcpp::export]]
List mprobit(List data) {

  arma::mat Y = data["Y"];
  arma::mat X = data["X"];
  arma::vec d = data["d"];
  int samples = data["samples"];
  int maxy = data["maxy"];

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
  arma::mat C(m, m);

  arma::vec r(m);

  arma::mat T(p, p);
  T = inv(X.t() * X + inv(arma::eye(p,p))); // note prior specification here

  arma::mat Bsave(samples, p * m, arma::fill::zeros);
  arma::mat Rsave(samples, m * (m - 1) / 2);

  double mj;
  arma::vec mi(m);
  arma::vec sj;
  arma::mat bj(m, m - 1);

  for (int k = 0; k < samples; ++k) {

    if ((k + 1) % 1000 == 0) {
      Rcpp::Rcout << "Sample: " << k + 1 << "\n";
    }

    // Sample latent responses.
    M = X * B;
    C = arma::chol(R, "lower");
    sj = sqrt(cdists(R));
    bj = cdistb(R);
    for (int i = 0; i < n; ++i) {
      if (std::isnan(d(i))) {
        for (int j = 0; j < m; ++j) {
          mj = cdistm(vectorise(M.row(i)), bj, vectorise(Z.row(i)), j);
          if (std::isnan(Y(i, j))) {
            Z(i, j) = R::rnorm(mj, sj(j));
          } else {
            Z(i, j) = rnormpos(mj, sj(j), Y(i, j) == 1);
          }
        }
      }
      else {
        mi = vectorise(M.row(i));
        do {
          Z.row(i) = mvrnorm(mi, C, true).t();
          for (int j = 0; j < m; ++j) {
            Y(i, j) = Z(i, j) > 0 ? 1 : 0;
          }
        } while (std::min(int(accu(Y.row(i))), maxy) != d(i));
      }
    }

    // Sample variances and covariances.
    r = diagvec(inv(R));
    for (int j = 0; j < m; ++j) {
      D(j, j) = 1 / sqrt(R::rgamma((m + 1) / 2.0, r(j) / 2.0));
    }
    W = Z * D;
    U = T * X.t() * W;
    S = inv(rwishart(n + 2, inv(W.t() * W + arma::eye(m, m) - U.t() * inv(T) * U)));

    // Sample beta parameters
    G = mvrnorm(U, T, S);

    // Standardize beta parameters and covariances into correlations.
    D = inv(sqrt(diagmat(S)));
    B = G * D;
    R = D * S * D;

    // Save sampled parameters.
    Bsave.row(k) = vectorise(B).t();
    Rsave.row(k) = lowertri(R, false).t();
  }

  return List::create(
    Named("beta") = wrap(Bsave),
    Named("corr") = wrap(Rsave)
  );
}
