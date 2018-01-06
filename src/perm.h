// Function declarations for perm.cpp. 

#ifndef PERM_H
#define PERM_H

#include <RcppArmadillo.h>

arma::vec bernperm(int samples, arma::vec y, arma::vec p);
arma::vec poisperm(int samples, arma::vec y, arma::vec lambda);
arma::vec normperm(int samples, arma::vec y, arma::vec mu, double sigm);

#endif