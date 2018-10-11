#include <RcppArmadillo.h>
#include "dist.h" // for mvrnorm, rnormpos
#include "post.h" // for posterior distribution samplers
#include "misc.h" // for lowertri

using namespace Rcpp;

// To do: Add prior specification as function arguments.

//' @export
// [[Rcpp::export]]
List raschic(arma::mat Y, arma::mat X, arma::mat Z, arma::vec d, int samples, int maxy) {

  int n = Y.n_rows;
  int m = Y.n_cols;
  int p = X.n_cols;
  int q = Z.n_cols;

  arma::vec u(n * m, arma::fill::zeros);

  arma::vec beta(p, arma::fill::zeros);
  arma::mat zeta(n, q, arma::fill::zeros);
  arma::mat phiv(q, q, arma::fill::eye);

  arma::mat bsave(samples, p, arma::fill::zeros);
  arma::mat vsave(samples, q * (q + 1) / 2, arma::fill::zeros);

  arma::vec mi(m);
  int lw, up;

  arma::vec mb(p, arma::fill::zeros);   // beta prior (location)
  arma::mat Rb(p, p, arma::fill::eye);  // beta prior (precision)

  for (int k = 0; k < samples; ++k) {

    if ((k + 1) % 1000 == 0) {
      Rcout << "Sample: " << k + 1 << "\n";
    }

    // sample latent responses
    for (int i = 0; i < n; ++i) {
      lw = i * m; up = lw + m - 1;
      mi = X.rows(lw, up) * beta + Z.rows(lw, up) * zeta.row(i).t();
      if (std::isnan(d(i))) {
        for (int j = 0; j < m; ++j) {
          if (std::isnan(Y(i,j))) {
            u(i * m + j) = R::rnorm(mi(j), 1.0);
          } else {
            u(i * m + j) = rnormpos(mi(j), 1.0, Y(i,j) == 1);
          }
        }
      }
      else {
        do {
          u.subvec(lw, up) = arma::randn(m) + mi;
          for (int j = 0; j < m; ++j) {
            Y(i,j) = u(lw + j) > 0 ? 1 : 0;
          }
        } while (std::min(static_cast<int>(accu(Y.row(i))), maxy) != d(i));
      }
    }

    // sample beta
    beta = betapost(X, Z, u, m, 1.0, phiv, mb, Rb);
    bsave.row(k) = beta.t();

    // sample respondent-specific parameters
    for (int i = 0; i < n; ++i) {
      lw = i * m; up = lw + m - 1;
      zeta.row(i) = betapost(Z.rows(lw,up), u.subvec(lw,up) -
        X.rows(lw,up) * beta, 1.0, arma::zeros(q), inv(phiv)).t();
    }

    // sample (co)variance of latent variable(s)
    if (q == 1) {
      phiv(0,0) = sigmpost(vectorise(zeta), 0.0, 10.0/2, 10.0/2); // note prior specification
    } else {
      phiv = covmpost(zeta, 10, arma::eye(q,q)/10); // note prior specification
    }
    vsave.row(k) = lowertri(phiv).t();
  }

  return List::create(
    Named("beta") = wrap(bsave),
    Named("sigm") = wrap(vsave)
  );
}
