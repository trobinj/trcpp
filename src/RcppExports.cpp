// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <Rcpp.h>

using namespace Rcpp;

// bernlong
List bernlong(arma::vec y, arma::mat x, arma::vec z, int m, arma::vec block, int samples, arma::mat betaprior, arma::vec phivprior, double betadelt, double zetadelt);
RcppExport SEXP _trcpp_bernlong(SEXP ySEXP, SEXP xSEXP, SEXP zSEXP, SEXP mSEXP, SEXP blockSEXP, SEXP samplesSEXP, SEXP betapriorSEXP, SEXP phivpriorSEXP, SEXP betadeltSEXP, SEXP zetadeltSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type z(zSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type block(blockSEXP);
    Rcpp::traits::input_parameter< int >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type betaprior(betapriorSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type phivprior(phivpriorSEXP);
    Rcpp::traits::input_parameter< double >::type betadelt(betadeltSEXP);
    Rcpp::traits::input_parameter< double >::type zetadelt(zetadeltSEXP);
    rcpp_result_gen = Rcpp::wrap(bernlong(y, x, z, m, block, samples, betaprior, phivprior, betadelt, zetadelt));
    return rcpp_result_gen;
END_RCPP
}
// foo
arma::mat foo(int n);
RcppExport SEXP _trcpp_foo(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(foo(n));
    return rcpp_result_gen;
END_RCPP
}
// lmerlong
List lmerlong(arma::vec y, arma::mat x, arma::mat z, int m, arma::vec block, arma::vec samples, arma::mat betaprior, arma::vec phivprior, arma::vec psivprior);
RcppExport SEXP _trcpp_lmerlong(SEXP ySEXP, SEXP xSEXP, SEXP zSEXP, SEXP mSEXP, SEXP blockSEXP, SEXP samplesSEXP, SEXP betapriorSEXP, SEXP phivpriorSEXP, SEXP psivpriorSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type z(zSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type block(blockSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type betaprior(betapriorSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type phivprior(phivpriorSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type psivprior(psivpriorSEXP);
    rcpp_result_gen = Rcpp::wrap(lmerlong(y, x, z, m, block, samples, betaprior, phivprior, psivprior));
    return rcpp_result_gen;
END_RCPP
}
// lmerperm
List lmerperm(arma::mat x, arma::mat z, arma::vec y, arma::vec clust, arma::vec block, arma::vec samples, arma::mat betaprior, arma::vec phivprior, arma::vec psivprior);
RcppExport SEXP _trcpp_lmerperm(SEXP xSEXP, SEXP zSEXP, SEXP ySEXP, SEXP clustSEXP, SEXP blockSEXP, SEXP samplesSEXP, SEXP betapriorSEXP, SEXP phivpriorSEXP, SEXP psivpriorSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type clust(clustSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type block(blockSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type betaprior(betapriorSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type phivprior(phivpriorSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type psivprior(psivpriorSEXP);
    rcpp_result_gen = Rcpp::wrap(lmerperm(x, z, y, clust, block, samples, betaprior, phivprior, psivprior));
    return rcpp_result_gen;
END_RCPP
}
// mprobit
List mprobit(arma::mat Y, arma::mat X, arma::vec d, int samples, int maxy);
RcppExport SEXP _trcpp_mprobit(SEXP YSEXP, SEXP XSEXP, SEXP dSEXP, SEXP samplesSEXP, SEXP maxySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< int >::type maxy(maxySEXP);
    rcpp_result_gen = Rcpp::wrap(mprobit(Y, X, d, samples, maxy));
    return rcpp_result_gen;
END_RCPP
}
// pmvnorm
double pmvnorm(arma::mat s, arma::vec a, arma::vec b, double epsi, double alph, int nmax);
RcppExport SEXP _trcpp_pmvnorm(SEXP sSEXP, SEXP aSEXP, SEXP bSEXP, SEXP epsiSEXP, SEXP alphSEXP, SEXP nmaxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type s(sSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a(aSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type epsi(epsiSEXP);
    Rcpp::traits::input_parameter< double >::type alph(alphSEXP);
    Rcpp::traits::input_parameter< int >::type nmax(nmaxSEXP);
    rcpp_result_gen = Rcpp::wrap(pmvnorm(s, a, b, epsi, alph, nmax));
    return rcpp_result_gen;
END_RCPP
}
// raschic
List raschic(arma::mat Y, arma::mat X, arma::mat Z, arma::vec d, int samples, int maxy);
RcppExport SEXP _trcpp_raschic(SEXP YSEXP, SEXP XSEXP, SEXP ZSEXP, SEXP dSEXP, SEXP samplesSEXP, SEXP maxySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< int >::type maxy(maxySEXP);
    rcpp_result_gen = Rcpp::wrap(raschic(Y, X, Z, d, samples, maxy));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_trcpp_bernlong", (DL_FUNC) &_trcpp_bernlong, 10},
    {"_trcpp_foo", (DL_FUNC) &_trcpp_foo, 1},
    {"_trcpp_lmerlong", (DL_FUNC) &_trcpp_lmerlong, 9},
    {"_trcpp_lmerperm", (DL_FUNC) &_trcpp_lmerperm, 9},
    {"_trcpp_mprobit", (DL_FUNC) &_trcpp_mprobit, 5},
    {"_trcpp_pmvnorm", (DL_FUNC) &_trcpp_pmvnorm, 6},
    {"_trcpp_raschic", (DL_FUNC) &_trcpp_raschic, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_trcpp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
