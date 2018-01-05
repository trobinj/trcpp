// Function declarations for logl.cpp. 

#ifndef _LOGL_H_
#define _LOGL_H_

#include <RcppArmadillo.h>

double normlogl(arma::vec y, arma::vec mu, double sigma);
double bernlogl(arma::vec y, arma::vec p);
double poislogl(arma::vec y, arma::vec lambda);

#endif