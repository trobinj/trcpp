// Functions to sample from the (conditional) posterior distribution 
// of the permutation of a vector of independent random variables. 

#include <RcppArmadillo.h>
#include "dist.h" // for randint
#include "misc.h" // for vswap

// Sampler for bernoulli-distributed variables.
arma::vec bernperm(int samples, arma::vec y, arma::vec p) {
  int n = y.n_elem;
  double lr;
  arma::ivec u(2);  
  for (int i = 0; i < samples; i++) {
    do {
      u = randint(2, 0, n - 1);  
    } while (u(0) == u(1));
    lr = R::dbinom(y(u(1)), 1, p(u(0)), true) + R::dbinom(y(u(0)), 1, p(u(1)), true) - 
      R::dbinom(y(u(0)), 1, p(u(0)), true) - R::dbinom(y(u(1)), 1, p(u(1)), true);
    if (R::runif(0.0, 1.0) < exp(lr)) {
      vswap(y, u(0), u(1));
    }
  }
  return y;
}

// Sampler for poisson-distributed variables.
arma::vec poisperm(int samples, arma::vec y, arma::vec lambda) {
  int n = y.n_elem;
  double lr;
  arma::ivec u(2);
  for (int i = 0; i < samples; i++) {
    do {
      u = randint(2, 0, n - 1);  
    } while (u(0) == u(1));
    lr = R::dpois(y(u(1)), lambda(u(0)), true) + R::dpois(y(u(0)), lambda(u(1)), true) - 
      R::dpois(y(u(0)), lambda(u(0)), true) - R::dpois(y(u(1)), lambda(u(1)), true);
    if (R::runif(0.0, 1.0) < exp(lr)) {
      vswap(y, u(0), u(1));
    }
  }
  return y;
}

// Sampler for normally-distributed variables.
arma::vec normperm(int samples, arma::vec y, arma::vec mu, double sigm) {
  int n = y.n_elem;
  double lr;
  arma::ivec u(2);  
  for (int i = 0; i < samples; i++) {
    do {
      u = randint(2, 0, n - 1);  
    } while (u(0) == u(1));
    lr = R::dnorm(y(u(1)), mu(u(0)), sigm, true) + R::dnorm(y(u(0)), mu(u(1)), sigm, true) - 
      R::dnorm(y(u(0)), mu(u(0)), sigm, true) - R::dnorm(y(u(1)), mu(u(1)), sigm, true);
    if (R::runif(0.0, 1.0) < exp(lr)) {
      vswap(y, u(0), u(1));
    }
  }
  return y;
}
