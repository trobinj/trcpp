// Function declarations for logl.cpp. 

#ifndef LOGL_H
#define LOGL_H

#include <RcppArmadillo.h>

double normlogl(arma::vec y, double mu, double sigma);
double normlogl(arma::vec y, arma::vec mu, double sigma);
double bernlogl(arma::vec y, arma::vec p);
double bernlogl(arma::vec y, double p);
double poislogl(arma::vec y, arma::vec lambda);
double poislogl(arma::vec y, double lambda);

#endif