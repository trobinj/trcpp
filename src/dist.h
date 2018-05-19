// Function declarations for dist.cpp. 

#ifndef DIST_H
#define DIST_H

#include <RcppArmadillo.h>

double rtnorm(double mu, double sigma, double a, double b);
double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logd);
arma::vec mvrnorm(arma::vec mu, arma::mat sigma);
arma::vec rmvt(arma::vec m, arma::mat s, double v);
double dmvt(arma::vec y, arma::vec m, arma::mat s, double v, bool logd);
double dwishart(arma::mat x, double n, arma::mat v, bool logd);
arma::mat rwishart(int df, arma::mat S);
arma::ivec randint(int n, int a, int b);

#endif