// Function declarations for dist.cpp. 

#ifndef _DIST_H_
#define _DIST_H_

#include <RcppArmadillo.h>

double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logdensity);
arma::vec mvrnorm(arma::vec mu, arma::mat sigma);
arma::mat rwishart(int df, arma::mat S);
arma::ivec randint(int n, int a, int b);

#endif