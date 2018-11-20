// Miscellaneous C++ functions using GSL functions. 

#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <gsl/gsl_integration.h>

// Compute nodes and weights for the Gauss-Hermite quadrature approximation
// \int_{-\infty}^{\infty} e^{-x^2} f(x) dx \approx \sum_{i=1}^n w_i f(x_i).
void ghquad(int n, arma::vec &node, arma::vec &wght) {
  
  const gsl_integration_fixed_type *t = gsl_integration_fixed_hermite;

  gsl_integration_fixed_workspace *w;
  w = gsl_integration_fixed_alloc(t, n, 0.0, 1.0, 0.0, 0.0);
  
  double *node_pntr = gsl_integration_fixed_nodes(w);
  double *wght_pntr = gsl_integration_fixed_weights(w);
  
  for (int i = 0; i < n; ++i) {
    node(i) = *(node_pntr + i);
    wght(i) = *(wght_pntr + i);
  }
}

/*
 * To do: Overload to create function to compute nodes and weights for integration
 * over a normal distribution with given mean and variance.
 */
