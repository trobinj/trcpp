#include <RcppArmadillo.h>
#include "dist.h"
#include "misc.h"

using namespace Rcpp;

namespace rnkspc {

  double pnorm(double x) {
    return R::pnorm(x, 0.0, 1.0, true, false);
  }

  double trnorm(double m, double s, double a, double b) {
    double y;
    do {
      y = R::rnorm(m, s);
    } while (a > y || y > b);
    return y;
  }

  // Sampler for truncated normal distribution with non-zero support
  // on (a,b) and (c,d) where a < b < c < d.
  double truncmix(double m, double s, double a, double b, double c, double d) {
    double pab = pnorm((b-m)/s) - pnorm((a-m)/s);
    double pcd = pnorm((d-m)/s) - pnorm((c-m)/s);
    if (R::runif(0.0, 1.0) < pab / (pab + pcd)) {
      return trnorm(m, s, a, b); 
    } else {
      return trnorm(m, s, c, d); 
    }
  }

  // Compute mean and variance of conditional distribution.
  void cdist(arma::vec m, arma::mat s, arma::vec x, int j, double & mj, double & vj) {

    arma::mat m2(m);
    m2.shed_row(j);

    arma::mat x2(x);
    x2.shed_row(j);

    arma::mat s22 = s;
    s22.shed_row(j);
    s22.shed_col(j);

    arma::mat s12 = s.row(j);
    s12.shed_col(j);

    arma::mat b = s12 * inv(s22);

    mj = m(j) + as_scalar(b * (x2 - m2));
    vj = s(j,j) - as_scalar(b * s12.t());
  }
}

//' @export
// [[Rcpp::export]]
List mnprnk(List data) {
  
  arma::mat Y = data["Y"];
  arma::mat X = data["X"];
  int samples = data["samples"];
  arma::vec beta = data["beta"];
  arma::mat S = data["S"];
  
  int n = Y.n_rows;
  int m = Y.n_cols;
  int p = X.n_cols;
  
  arma::mat W = inv(S);
  arma::mat M(n, m);
  
  arma::mat XWX(p, p);
  arma::vec XWu(p);
  arma::mat Xi(m, p);
  arma::vec ui(m);
  arma::vec mi(m);
  arma::vec yi(m);
  arma::mat B(p, p);
  arma::mat Z(n, m);
  arma::mat U(n, m);
    
  // Prior parameter specification.
  arma::mat Rb = arma::eye(p,p);
  arma::mat V = arma::eye(m,m) * 10;
  int v = 10;
  
  double alph;

  for (int i = 0; i < n; ++i) {
    mi = X.rows(i * m, i * m + m - 1) * beta;
    M.row(i) = mi.t();
  }
    
  // Initialize latent responses, consistent with observed rankings.
  for (int i = 0; i < n; ++i) {
    ui.randn();
    arma::vec utmp(m, arma::fill::randn);
    utmp = utmp(sort_index(abs(utmp)));
    for (int j = 0; j < m; ++j) {
      U(i,j) = utmp(Y(i,j) - 1);
    }
  }

  // Saved simulated realizations from posterior distribution.
  arma::mat betasave(samples, p);
  arma::mat sigmsave(samples, m * (m + 1) / 2);
  
  for (int k = 0; k < samples; ++k) {
    
    if ((k + 1) % 100 == 0) {
      Rcpp::Rcout << "Sample: " << k + 1 << "\n";
      Rcpp::checkUserInterrupt(); // for debugging
    }
    
    // Need to make truncated normal sampler more robust.
    
    for (int i = 0; i < n; ++i) {
      ui = vectorise(U.row(i));
      yi = vectorise(Y.row(i));
      mi = vectorise(M.row(i));
      for (int j = 0; j < m; ++j) {
        double lw, up;
        double mj, vj;
        if (Y(i,j) == 1) {
          lw = 0.0;
          up = as_scalar(abs(ui(find(yi == 2))));
        } else if (Y(i,j) == m) {
          lw = as_scalar(abs(ui(find(yi == m - 1))));
          up = 1.0e+5;
        } else {
          lw = as_scalar(abs(ui(find(yi == yi(j) - 1))));
          up = as_scalar(abs(ui(find(yi == yi(j) + 1))));
        }
        rnkspc::cdist(mi, S, ui, j, mj, vj);
        ui(j) = rnkspc::truncmix(mj, sqrt(vj), -up, -lw, lw, up);
      }
      U.row(i) = ui.t();
    }
    
    // Sample beta.


    XWX.fill(0.0);
    XWu.fill(0.0);
    for (int i = 0; i < n; ++i) {
      Xi = X.rows(i * m, i * m + m - 1);
      ui = vectorise(U.row(i));
      XWX = XWX + Xi.t() * W * Xi;
      XWu = XWu + Xi.t() * W * ui;
    }
    B = inv(XWX + Rb);
    beta = mvrnorm(B * XWu, B, false);
       
    for (int i = 0; i < n; ++i) {
      mi = X.rows(i * m, i * m + m - 1) * beta;
      M.row(i) = mi.t();
    }

    // This tends to improve convergence. 
    if (k > 99) {
      Z = U - M;
      W = rwishart(n + v, inv(V + Z.t() * Z));
      S = inv(W);
    }

    alph = pow(det(S), 1.0 / m);
    betasave.row(k) = beta.t() / sqrt(alph);
    sigmsave.row(k) = lowertri(S, true).t() / alph;

    Rcpp::Rcout << "beta:\n" << beta / sqrt(alph) << "\n"; // for debugging
    Rcpp::Rcout << "sigm:\n" << S / alph << "\n";          // for debugging
  }
  
  return List::create(
    Named("sigm") = wrap(sigmsave),
    Named("beta") = wrap(betasave)
  );
}


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
