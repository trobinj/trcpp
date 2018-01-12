// Function declarations for post.cpp. 

#ifndef POST_H
#define POST_H

#include <RcppArmadillo.h>

arma::vec meanpost(arma::mat y, arma::mat sigma, arma::vec mu0, arma::mat sigma0);
arma::mat covmpost(arma::mat y, arma::vec mu, int df, arma::mat scale);
arma::vec betapost(arma::mat x, arma::vec y, double phiv, arma::vec mb, arma::mat Rb);
arma::vec betablockpost(arma::mat x, arma::mat z, arma::vec y, arma::vec clust, double psiv, arma::mat Rz, arma::vec mb, arma::mat Rb);
double sigmpost(arma::vec y, arma::vec mu, double alph, double beta);
arma::vec betapost2(arma::mat x, arma::vec y, arma::vec phiv, arma::vec mb, arma::mat Rb);
arma::vec betablockpost2(arma::mat x, arma::mat z, arma::vec y, arma::vec clust, arma::vec psiv, arma::mat Rz, arma::vec mb, arma::mat Rb);

#endif