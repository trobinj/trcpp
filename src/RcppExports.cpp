// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// lowertri
arma::vec lowertri(arma::mat x);
RcppExport SEXP _trcpp_lowertri(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(lowertri(x));
    return rcpp_result_gen;
END_RCPP
}
// meanpost
arma::vec meanpost(arma::mat y, arma::mat sigma, arma::vec mu0, arma::mat sigma0);
RcppExport SEXP _trcpp_meanpost(SEXP ySEXP, SEXP sigmaSEXP, SEXP mu0SEXP, SEXP sigma0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type sigma0(sigma0SEXP);
    rcpp_result_gen = Rcpp::wrap(meanpost(y, sigma, mu0, sigma0));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _trcpp_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _trcpp_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _trcpp_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _trcpp_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_trcpp_lowertri", (DL_FUNC) &_trcpp_lowertri, 1},
    {"_trcpp_meanpost", (DL_FUNC) &_trcpp_meanpost, 4},
    {"_trcpp_rcpparma_hello_world", (DL_FUNC) &_trcpp_rcpparma_hello_world, 0},
    {"_trcpp_rcpparma_outerproduct", (DL_FUNC) &_trcpp_rcpparma_outerproduct, 1},
    {"_trcpp_rcpparma_innerproduct", (DL_FUNC) &_trcpp_rcpparma_innerproduct, 1},
    {"_trcpp_rcpparma_bothproducts", (DL_FUNC) &_trcpp_rcpparma_bothproducts, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_trcpp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
