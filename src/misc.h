// Function declarations for misc.cpp. 

#ifndef MISC_H
#define MISC_H

#include <RcppArmadillo.h>

void vswap(arma::vec & x, int a, int b);
void vswap(arma::ivec & x, int a, int b);
arma::vec repeat(arma::vec x, int n);
arma::vec repeat(arma::vec x, arma::vec n);
arma::vec lowertri(arma::mat x);
arma::mat vec2symm(arma::vec x);
arma::vec invlogit(arma::vec x);
arma::umat indexmat(arma::vec x);

#endif