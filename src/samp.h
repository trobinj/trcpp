// Function declarations for misc.cpp. 

#ifndef SAMP_H
#define SAMP_H

#include <RcppArmadillo.h>

arma::vec rnormsum(arma::vec mu, arma::vec sigma, double t, int n, double delta);

#endif