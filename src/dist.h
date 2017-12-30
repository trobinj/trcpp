#ifndef DIST_H
#define DIST_H

#include <RcppArmadillo.h>

double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logdensity);

arma::vec mvrnorm(arma::vec mu, arma::mat sigma);

#endif