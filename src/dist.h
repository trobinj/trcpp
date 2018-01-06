// Function declarations for dist.cpp. 

#ifndef DIST_H
#define DIST_H

#include <RcppArmadillo.h>

double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logdensity);
arma::vec mvrnorm(arma::vec mu, arma::mat sigma);
arma::mat rwishart(int df, arma::mat S);
arma::ivec randint(int n, int a, int b);

#endif