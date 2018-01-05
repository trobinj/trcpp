// Function declarations for misc.cpp. 

#ifndef _MISC_H_
#define _MISC_H_

#include <RcppArmadillo.h>

arma::vec lowertri(arma::mat x);
arma::vec invlogit(arma::vec x);

#endif