// Function declarations for dist.cpp.

#ifndef DIST_H
#define DIST_H

#include <RcppArmadillo.h>

arma::ivec randint(int n, int a, int b);
int randint(int a, int b);
int rdiscrete(arma::vec wght);
void shuffle(arma::vec & x);
arma::vec srs(arma::vec x, int n);

double rtnorm(double mu, double sigma, double a, double b);
double rnormpos(double m, double s, bool pos);
double rnormint(double m, double s, double a, double b);
double rnormtail(double a, double m, double s, bool pos);

double dmvnorm(arma::vec y, arma::vec mu, arma::mat sigma, bool logd);
arma::vec mvrnorm(arma::vec mu, arma::mat sigma, bool cholesky = false);
arma::mat mvrnorm(arma::mat m, arma::mat u, arma::mat v);

arma::vec rmvt(arma::vec m, arma::mat s, double v);
double dmvt(arma::vec y, arma::vec m, arma::mat s, double v, bool logd);

double dwishart(arma::mat x, double n, arma::mat v, bool logd);
arma::mat rwishart(int df, arma::mat S);

#endif
