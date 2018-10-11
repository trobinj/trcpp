// Source file for functions for sampling from a posterior distribution.

#include <RcppArmadillo.h>
#include "dist.h"
#include "misc.h"
#include "logl.h"

arma::vec betalogrpost(arma::mat x, arma::vec y, arma::vec z,
  arma::vec bold, arma::vec bm, arma::mat bs, double delt) {

  double numlogdens, denlogdens;
  int p = x.n_cols;

  arma::vec bnew(p);
  bnew = mvrnorm(bold, arma::eye(p,p) * delt);

  numlogdens = bernlogl(y, invlogit(x * bnew + z)) + dmvnorm(bnew, bm, bs, true);
  denlogdens = bernlogl(y, invlogit(x * bold + z)) + dmvnorm(bold, bm, bs, true);

  if (R::runif(0.0, 1.0) < exp(numlogdens - denlogdens)) {
    return bnew;
  } else {
    return bold;
  }
}

// Note: Sigma and sigma0 are variances here.
double meanpost(arma::vec y, double sigma, double mu0, double sigma0) {
  int n = y.n_elem;
  double vpost = 1.0 / (1.0 / sigma0 + n / sigma);
  double mpost = (mu0 / sigma0 + accu(y) / sigma) * vpost;
  return R::rnorm(mpost, sqrt(vpost));
}

arma::vec meanpost(arma::mat y, arma::mat sigma, arma::vec mu0, arma::mat sigma0) {
  int n = y.n_rows;
  int m = y.n_cols;
  arma::mat sigma0inv(m,m);
  arma::mat sigmainv(m,n);
  arma::mat B(m,m);
  arma::vec b(m);
  sigma0inv = arma::inv(sigma0);
  sigmainv = arma::inv(sigma);
  B = arma::inv(sigma0inv + n * sigmainv);
  b = sigma0inv * mu0 + n * sigmainv * arma::mean(y).t();
  return mvrnorm(B * b, B);
}

// Note: Here alph and beta are the shape and rate parameters for the
// parameterization of the gamma distribution (i.e., 1/beta = scale).
// Also note that the second argument of R::rgamma is scale, not rate.

double sigmpost(arma::vec y, arma::vec mu, double alph, double beta) {
  int n = y.n_elem;
  double a = alph + n / 2.0;
  double b = beta + accu(square(y - mu)) / 2.0;
  return 1 / R::rgamma(a, 1/b);
}

double sigmpost(arma::vec y, double mu, double alph, double beta) {
  int n = y.n_elem;
  double a = alph + n / 2.0;
  double b = beta + accu(square(y - mu)) / 2.0;
  return 1 / R::rgamma(a, 1/b);
}

arma::mat covmpost(arma::mat y, arma::vec mu, int df, arma::mat scale) {
  int n = y.n_rows;
  int m = y.n_cols;
  arma::mat v(m, m, arma::fill::zeros);
  arma::vec z(m);
  for (int i = 0; i < n; ++i) {
    z = y.row(i).t() - mu;
    v = v + z * z.t();
  }
  return inv(rwishart(n + df, inv(inv(scale) + v)));
}

arma::mat covmpost(arma::mat y, int df, arma::mat scale) {
  int n = y.n_rows;
  int m = y.n_cols;
  arma::mat v(m, m, arma::fill::zeros);
  arma::vec z(m);
  for (int i = 0; i < n; ++i) {
    z = y.row(i).t();
    v = v + z * z.t();
  }
  return inv(rwishart(n + df, inv(inv(scale) + v)));
}

arma::vec betapost(arma::mat x, arma::vec y, double psiv, arma::vec mb, arma::mat Rb) {
  int p = x.n_cols;
  arma::mat B(p, p);
  arma::vec b(p);
  B = inv(x.t() * x / psiv + Rb);
  b = x.t() * y / psiv + Rb * mb;
  return mvrnorm(B * b, B);
}

// Sample beta from the posterior distribution of a normal linear mixed
// model, assuming simple error variance structure and equal-sized clusters.
arma::vec betapost(arma::mat x, arma::mat z, arma::vec y, int m,
  double psiv, arma::mat phiv, arma::vec mb, arma::mat Rb) {
  int n = x.n_rows/m;
  int p = x.n_cols;
  int q = z.n_cols;
  int lw, up;
  arma::mat xwx(p, p, arma::fill::zeros);
  arma::vec xwy(p, arma::fill::zeros);
  arma::mat xw(p,m);
  arma::mat xi(m,p);
  arma::mat zi(m,q);
  arma::vec yi(m);
  arma::mat B(p,p);
  arma::vec b(p);
  for (int i = 0; i < n; ++i) {
    lw = i * m; up = lw + m - 1;
    xi = x.rows(lw,up);
    zi = z.rows(lw,up);
    yi = y.subvec(lw,up);
    xw = xi.t() * inv(zi * phiv * zi.t() + arma::eye(m,m) * psiv);
    xwx = xwx + xw * xi;
    xwy = xwy + xw * yi;
  }
  B = inv(xwx + Rb);
  b = xwy + Rb * mb;
  return mvrnorm(B * b, B);
}

arma::vec betablockpost(arma::mat x, arma::mat z, arma::vec y, arma::vec clust,
  double psiv, arma::mat phiv, arma::vec mb, arma::mat Rb) {

  int n = max(clust);
  int p = x.n_cols;
  int q = z.n_cols;

  arma::mat xwx(p, p, arma::fill::zeros);
  arma::vec xwy(p, arma::fill::zeros);

  arma::mat B(p, p);
  arma::vec b(p);

  arma::umat indx = indexmat(clust);
  unsigned int low, upp;
  int m;

  for (int i = 0; i < n; ++i) {

    low = indx(i, 0);
    upp = indx(i, 1);
    m = upp - low + 1;

    arma::mat xi(m, p);
    arma::mat zi(m, q);
    arma::vec yi(m);
    arma::mat xw(p, m);

    xi = x.rows(low, upp);
    zi = z.rows(low, upp);
    yi = y(arma::span(low, upp));

    xw = xi.t() * inv(zi * phiv * zi.t() + arma::eye(m, m) * psiv);
    xwx = xwx + xw * xi;
    xwy = xwy + xw * yi;
  }

  B = inv(xwx + Rb);
  b = xwy + Rb * mb;

  return mvrnorm(B * b, B);
}
