// Function declarations for misc.cpp. 

#ifndef MISC_H
#define MISC_H

#include <RcppArmadillo.h>

arma::vec lowertri(arma::mat x);
arma::mat vec2symm(arma::vec x);
arma::vec invlogit(arma::vec x);
arma::umat indexmat(arma::vec x);

#endif