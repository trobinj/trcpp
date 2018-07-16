// Function declarations for tilt.cpp. 

#ifndef TILT_H
#define TILT_H

#include <RcppArmadillo.h>

arma::vec bernrjct(arma::vec prb, int s);
arma::vec multrjct(arma::mat prb, arma::vec s);

#endif