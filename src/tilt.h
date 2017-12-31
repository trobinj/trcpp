// Function declarations for tilt.cpp. 

#ifndef _TILT_H_
#define _TILT_H_

#include <RcppArmadillo.h>

double bernscor(arma::vec prb, double t, int s);
double bernroot(arma::vec prb, int s, double a, double b, double n);
arma::vec bernrjct(arma::vec prb, int s);

#endif