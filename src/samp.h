// Function declarations for misc.cpp. 

#ifndef SAMP_H
#define SAMP_H

#include <RcppArmadillo.h>

arma::vec varmsamp(arma::vec y, arma::mat x, arma::mat z, double m, 
  arma::vec beta, arma::vec mb, arma::mat Rb, 
  arma::vec vold, arma::vec vm, arma::mat vs,
  arma::vec phivprior, arma::vec psivprior);

#endif