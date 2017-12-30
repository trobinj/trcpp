// Function declarations for perm.cpp. 

#ifndef _PERM_H_
#define _PERM_H_

#include <RcppArmadillo.h>

arma::vec bernperm(int samples, arma::vec y, arma::vec p);
arma::vec poisperm(int samples, arma::vec y, arma::vec lambda);
arma::vec normperm(int samples, arma::vec y, arma::vec mu, double sigm);

#endif