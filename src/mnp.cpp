#include <RcppArmadillo.h>
#include "dist.h"
#include "misc.h"

using namespace Rcpp;

namespace mnpspc {

  double pnorm(double x) {
    return R::pnorm(x, 0.0, 1.0, true, false);
  }

  // Sampler for truncated normal distribution with non-zero support on (a,b) and (c,d) where a < b < c < d.
  double truncmix(double m, double s, double a, double b, double c, double d) {
    double pab = pnorm((b - m) / s) - pnorm((a - m) / s);
    double pcd = pnorm((d - m) / s) - pnorm((c - m) / s);
    if (R::runif(0.0, 1.0) < pab / (pab + pcd)) {
      return rnormint(m, s, a, b);
    } else {
      return rnormint(m, s, c, d);
    }
  }

  arma::mat bcoef(arma::mat s) {
    int m = s.n_cols;
    arma::mat y(m, m - 1);

    for (int j = 0; j < m; ++j) {

      arma::mat s12 = s.row(j);
      arma::mat s22 = s;

      s12.shed_col(j);
      s22.shed_row(j);
      s22.shed_col(j);

      y.row(j) = s12 * inv_sympd(s22);
    }
    return y;
  }

  void cdist(arma::vec m, arma::mat s, arma::mat b, arma::vec x, int j, double & mj, double & vj) {
      arma::mat m2(m); m2.shed_row(j);
      arma::mat x2(x); x2.shed_row(j);
      arma::mat s12 = s.row(j); s12.shed_col(j);

      mj = m(j) + as_scalar(b.row(j) * (x2 - m2));
      vj = s(j,j) - as_scalar(b.row(j) * s12.t());
  }
}

/* Note: This algorithm uses a method from Burgette and Nordheim 
(2012, Journal of Business and Economic Statistics) to impose an
identification constraint on the trace of the covariance matrix. 
To improve the behavior of the MCMC algorithm, sampling of the
covariance matrix is delayed until after a short burn-in. */

//' @export
// [[Rcpp::export]]
List mnprnk(List data) {

  using namespace mnpspc;

  arma::mat Y = data["Y"];
  arma::mat X = data["X"];
  int samples = data["samples"];
  arma::vec beta = data["beta"];
  arma::mat S = data["S"];
  
  arma::vec mb = data["mb"];
  arma::mat Rb = data["Rb"];

  int n = Y.n_rows;
  int m = Y.n_cols;
  int p = X.n_cols;

  arma::mat W = inv_sympd(S);
  arma::mat M(n, m);
  arma::mat C(m, m);

  arma::mat XWX(p, p);
  arma::vec XWu(p);
  arma::mat Xi(m, p);
  arma::vec ui(m);
  arma::vec mi(m);
  arma::vec yi(m);
  arma::mat B(p, p);
  arma::vec b(p);
  arma::mat Z(n, m);
  arma::mat U(n, m);

  arma::vec zeta(n);
  
  double a2 = 1.0;
  double a1 = sqrt(a2);
  double ssa = 0.0;
  
  arma::mat V = arma::eye(m,m); // prior scale matrix
  int v = m + 1;                // prior degrees of freedom

  for (int i = 0; i < n; ++i) {
    mi = X.rows(i * m, i * m + m - 1) * beta;
    M.row(i) = mi.t();
  }

  // Initialize latent responses that are consistent with observed rankings.
  for (int i = 0; i < n; ++i) {
    ui.randn();
    ui = ui(sort_index(abs(ui)));
    for (int j = 0; j < m; ++j) {
      if (Y(i,j) == 0) {
        U(i,j) = R::rnorm(0.0, 1.0);
      }
      else {
        U(i,j) = ui(Y(i,j) - 1);
     }
    }
  }

  // Arrays to store stimulated realizations from the posterior distribution.
  arma::mat betasave(samples, p);
  arma::mat sigmsave(samples, m * (m + 1) / 2);
  arma::mat deltsave(samples, m);

  for (int k = 0; k < samples; ++k) {

    if ((k + 1) % 100 == 0) {
      Rcpp::Rcout << "Sample: " << k + 1 << "\n";
      Rcpp::checkUserInterrupt();
    }

    arma::mat Bk(m, m - 1);
    Bk = bcoef(S);

    for (int i = 0; i < n; ++i) {
      ui = vectorise(U.row(i));
      yi = vectorise(Y.row(i));
      mi = vectorise(M.row(i));
      for (int j = 0; j < m; ++j) {
        double lw, up;
        double mj, vj;
        if (Y(i,j) == 0) {
          lw = 0.0;
          up = 1.0e+5;
        } else if (Y(i,j) == 1) {
          lw = 0.0;
          up = min(abs(ui(find(yi > yi(j) && yi > 0))));
        } else if (Y(i,j) == max(yi)) {
          lw = max(abs(ui(find(yi < yi(j) && yi > 0))));
          up = 1.0e+5;
        } else {
          lw = max(abs(ui(find(yi < yi(j) && yi > 0))));
          up = min(abs(ui(find(yi > yi(j) && yi > 0))));
        }
        cdist(mi, S, Bk, ui, j, mj, vj);
        ui(j) = truncmix(mj, sqrt(vj), -up, -lw, lw, up);
      }
      U.row(i) = ui.t();
    }
    
    if (k > 999) {
      a2 = trace(V * W) / R::rchisq(v * m);
      a1 = sqrt(a2);
    }
    U = U * a1;
    
    // Sample beta.

    XWX.fill(0.0);
    XWu.fill(0.0);
    for (int i = 0; i < n; ++i) {
      Xi = X.rows(i * m, i * m + m - 1);
      ui = vectorise(U.row(i));
      XWX = XWX + Xi.t() * W * Xi;
      XWu = XWu + Xi.t() * W * ui;
    }
    B = inv_sympd(Rb + XWX);
    b = B * (Rb * mb + XWu);
  
    if (k > 999) {
      ssa = 0.0;
      for (int i = 0; i < n; ++i) {
        Xi = X.rows(i * m, i * m + m - 1);
        ui = vectorise(U.row(i));
        ssa = ssa + as_scalar((ui - Xi*b).t() * W * (ui - Xi*b));
      }
      a2 = (ssa + as_scalar(b.t() * Rb * b) + trace(V * W)) / R::rchisq((n + v) * m);
      a1 = sqrt(a2);  
    }
    
    beta = mvrnorm(b, a2 * B, false);
    
    betasave.row(k) = beta.t();

    for (int i = 0; i < n; ++i) {
      mi = X.rows(i * m, i * m + m - 1) * beta;
      M.row(i) = mi.t();
    }

    if (k > 999) {
      Z = U - M;
      W = rwishart(n + v, inv_sympd(V + Z.t() * Z));
      S = inv_sympd(W);

      a2 = trace(inv_sympd(W)) / m;
      a1 = sqrt(a2);
    }
    
    S = inv_sympd(W) / a2;
    W = inv_sympd(S);
    
    U = U / a1;
    M = M / a1;
    
    beta = beta / a1;

    // Compute sampled correlations instead of covariances.
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        if (i == j) {
          C(i,j) = S(i,j);  
        } else {
          C(i,j) = S(i,j) / sqrt(S(i,i) * S(j,j));  
        }
      }
    }

    sigmsave.row(k) = lowertri(C, true).t();
    deltsave.row(k) = mvrnorm(beta.head(m), S, false).t(); // posterior predictive distribution for "moderate"

    // Rcpp::Rcout << "beta:\n" << beta / sqrt(alph) << "\n";
    // Rcpp::Rcout << "sigm:\n" << S / alph << "\n";
  }

  return List::create(
    Named("sigm") = wrap(sigmsave),
    Named("beta") = wrap(betasave),
    Named("delt") = wrap(deltsave)
  );
}

/*
List mnpirt(List data) {

  using namespace mnpspc;

  arma::mat Y = data["Y"];          // maybe change type to unsigned integer (umat)
  arma::mat X = data["X"];
  arma::mat Z = data["Z"];
  int samples = data["samples"];
  int r = data["r"];                // number of stimuli per presentation

  int n = Y.n_rows;                 // number of respondents
  int m = Y.n_cols / r;             // number of presentations
  int c = m * r;                    // number of latent responses
  int p = X.n_cols;                 // number of covariate parameters
  int q = Z.n_cols;                 // number of respondent-specific parameters

  arma::mat M(n,c);
  arma::mat U(n,c);
  arma::vec mi(c);
  arma::vec ui(c);
  arma::vec yi(c);

  arma::mat XRX(p, p);
  arma::vec XRu(p);
  arma::mat Xi(c, p);
  arma::mat Zi(c, q);

  arma::mat Bb(p, p);
  arma::mat Bz(q, q);

  arma::vec beta(p, arma::fill::zeros);
  arma::mat Sz(q, q, arma::fill::eye);
  arma::mat Rz = inv(Sz);
  arma::mat Su(c, c, arma::fill::eye);
  arma::mat Ru = inv(Su);
  arma::mat zeta(n,q);

  // Prior parameter specification.
  arma::mat Rb = arma::eye(p,p);
  arma::mat Vz = arma::eye(q,q);
  int vz = q + 1;

  double alph = 1.0;

  arma::mat betasave(samples, p);
  arma::mat sigmsave(samples, c * (c + 1) / 2);

  for (int i = 0; i < n; ++i) {
    mi = X.rows(i * c, i * c + c - 1) * beta;
    M.row(i) = mi.t();
  }

  arma::vec yij(r);
  arma::vec uij(r);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      uij.randn();
      uij = uij(sort_index(abs(uij)));
      for (int k = 0; k < r; ++k) {
        if (Y(i, j * r + k) == 0) {
          U(i, j * r + k) = R::rnorm(0.0, 1.0);
        } else {
          U(i, j * r + k) = uij(Y(i, j * r + k) - 1);
        }
      }
    }
  }

  for (int t = 0; t < samples; ++t) {

    if ((t + 1) % 10 == 0) {
      Rcpp::Rcout << "Sample: " << t + 1 << "\n";
      Rcpp::checkUserInterrupt();
    }

    // Sample latent responses.

    for (int i = 0; i < n; ++i) {

      mi = vectorise(M.row(i));
      ui = vectorise(U.row(i));
      yi = vectorise(Y.row(i));

      for (int j = 0; j < m; ++j) {

        uij = ui(arma::span(j*r, j*r + r - 1));
        yij = yi(arma::span(j*r, j*r + r - 1));

        for (int k = 0; k < r; ++k) {

          double low, upp;
          double mjk, vjk;

          // might remove the zero-check --- not sure if it'll be used

          if (Y(i, j*r + k) == 0) {
            low = 0.0;
            upp = 1.0e+5;
          } else if (Y(i, j*r + k) == 1) {
            low = 0.0;
            upp = min(abs(uij(find(yij > yij(k) && yij > 0))));
          } else if (Y(i, j*r + k) == max(yij)) {
            low = max(abs(uij(find(yij < yij(k) && yij > 0))));
            upp = 1.0e+5;
          } else {
            low = max(abs(uij(find(yij < yij(k) && yij > 0))));
            upp = min(abs(uij(find(yij > yij(k) && yij > 0))));
          }

          cdist(mi, Su, ui, j*r + k, mjk, vjk);
          ui(j * r + k) = truncmix(mjk, sqrt(vjk), -upp, -low, low, upp);
          uij(k) = ui(j*r + k);
        }
      }
      U.row(i) = ui.t();
    }

    // Sample beta.

    XRX.fill(0.0);
    XRu.fill(0.0);
    for (int i = 0; i < n; ++i) {
      Xi = X.rows(i * c, i * c + c - 1);
      ui = vectorise(U.row(i));
      XRX = XRX + Xi.t() * Ru * Xi;
      XRu = XRu + Xi.t() * Ru * ui;
    }
    Bb = inv_sympd(XRX + Rb);
    beta = mvrnorm(Bb * XRu, Bb, false);

    for (int i = 0; i < n; ++i) {
      mi = X.rows(i * c, i * c + c - 1) * beta;
      M.row(i) = mi.t();
    }

    // Sample zeta.

    for (int i = 0; i < n; ++i) {
      Zi = Z.rows(i * c, i * c + c - 1);
      ui = vectorise(U.row(i)) - X.rows(i * c, i * c + c - 1) * beta;
      Bz = inv_sympd(Zi.t() * Zi + Rz);
      zeta.row(i) = mvrnorm(Bz * Zi.t() * ui, Bz, false).t();
    }

    // Sample phi matrix.

    // Delay sampling from conditional posterior of covariance matrix to improve initial mixing.
    if (t > 499) {
      Rz = rwishart(n + vz, inv_sympd(Vz + zeta.t() * zeta));
      Sz = inv(Rz);
    }

    Zi = Z.rows(0, c - 1); // assuming same covariance structure for all i
    Su = Zi * Sz * Zi.t() + arma::eye(c,c);
    Ru = inv_sympd(Su);

    // Store sample. Here alpha is a standardization constant (not used).

    betasave.row(t) = beta.t() / sqrt(alph);
    sigmsave.row(t) = lowertri(Su, true).t() / alph;
  }

  return List::create(
    Named("beta") = wrap(betasave),
    Named("sigm") = wrap(sigmsave)
  );
}
*/

/*
List mnprnk(List data) {

  arma::mat Y = data["Y"];
  arma::mat X = data["X"];
  int samples = data["samples"];

  int n = Y.n_rows;
  int m = Y.n_cols;
  int p = X.n_cols;

  // Parameters with initial values.
  arma::vec beta(p, arma::fill::zeros);
  arma::mat S(m, m, arma::fill::eye);

  // Functions of parameters.
  arma::mat W = inv(S);
  arma::vec bt(m);
  double a2 = trace(S)/m;

  // Prior parameter specification.
  arma::mat Rb = arma::eye(p,p);
  arma::mat V = arma::eye(m,m) / 1000;
  int v = 1000;

  // Saved simulated realizations from posterior distribution.
  arma::mat betasave(samples, p);
  arma::mat sigmsave(samples, m * (m + 1) / 2);

  // Initialize latent responses, consistent with observed rankings.
  arma::mat U(size(Y));
  for (int i = 0; i < n; ++i) {
    arma::vec utmp(m, arma::fill::randn);
    utmp = utmp(sort_index(abs(utmp)));
    for (int j = 0; j < m; ++j) {
      U(i,j) = utmp(Y(i,j) - 1);
    }
  }
  arma::mat Ut(size(U));

  for (int k = 0; k < samples; ++k) {

    Rcpp::checkUserInterrupt();
    if ((k + 1) % 1000 == 0) {
      Rcpp::Rcout << "Sample: " << k + 1 << "\n";
    }

    // Step 1: Sample w and a2.

    // Sample latent responses.
    for (int i = 0; i < n; ++i) {
      arma::rowvec ui = U.row(i);
      arma::rowvec yi = Y.row(i);
      for (int j = 0; j < m; ++j) {
        double lw, up;
        if (Y(i,j) == 1) {
          lw = 0.0;
          up = as_scalar(abs(ui(find(yi == 2))));
          // put sampler here?
        } else if (Y(i,j) == m) {
          lw = as_scalar(abs(ui(find(yi == m - 1))));
          up = 1.0e+5;
          // put sampler here?
        } else {
          lw = as_scalar(abs(ui(find(yi == yi(j) - 1))));
          up = as_scalar(abs(ui(find(yi == yi(j) + 1))));
          // put sampler here?
        }
        arma::vec mi = X.rows(i * m, i * m + m - 1) * beta;
        double mj, vj;
        rnkspc::cdist(mi, S, vectorise(ui), j, mj, vj);
        ui(j) = rnkspc::truncmix(mj, sqrt(vj), -up, -lw, lw, up);
      }
      U.row(i) = ui;
    }

    // Sampling a2 and computing transformed latent responses.
    // a2 = trace(V * W) / R::rchisq(v * m);
    Ut = U * sqrt(a2);

    // Step 2: Sample beta and a2.

    {
      arma::mat XWX(p, p, arma::fill::zeros);
      arma::vec XWu(p, arma::fill::zeros);
      for (int i = 0; i < n; ++i) {
        arma::mat Xi = X.rows(i * m, i * m + m - 1);
        arma::vec ui = vectorise(Ut.row(i));
        XWX = XWX + Xi.t() * W * Xi;
        XWu = XWu + Xi.t() * W * ui;
      }
      arma::vec b = inv(XWX + Rb) * XWu;

      double ss = 0.0;
      for (int i = 0; i < n; ++i) {
        arma::mat Xi = X.rows(i * m, i * m + m - 1);
        arma::vec ui = vectorise(Ut.row(i));
        ss = ss + as_scalar((ui - Xi*b).t() * W * (ui - Xi*b));
      }
      a2 = (ss + as_scalar(b.t() * Rb * b) + trace(V * W)) / R::rchisq((n + v) * (m - 1));

      bt = mvrnorm(b, a2 * inv(XWX + Rb));
      betasave.row(k) = bt.t() / sqrt(a2);
      beta = bt / sqrt(a2);
    }

    // Step 3: Sampling S and scaling.

    if (true) {
      arma::mat St(size(S));
      arma::mat SS(m, m, arma::fill::zeros);
      for (int i = 0; i < n; ++i) {
        arma::mat Xi = X.rows(i * m, i * m + m - 1);
        arma::vec ui = vectorise(Ut.row(i));
        SS = SS + (ui - Xi*bt) * (ui - Xi*bt).t();
      }
      St = inv(rwishart(n + v, inv(S + SS)));

      a2 = trace(St) / m;
      a2 = St(0,0);

      S = St / a2;

      W = inv(S);
      U = Ut / sqrt(a2);
      //beta = bt / sqrt(a2);

      sigmsave.row(k) = lowertri(S, true).t();
    }

    Rcpp::Rcout << beta << "\n";
    Rcpp::Rcout << S << "\n";
  }

  return List::create(
    Named("sigm") = wrap(sigmsave),
    Named("beta") = wrap(betasave)
  );
}
 */
