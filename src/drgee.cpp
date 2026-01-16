#include <RcppArmadillo.h>

// using namespace Rcpp;
// using namespace std;
// using namespace arma;

////////////////////// Residuals from conditional logistic regression //////////////////


void conditResClust(const arma::vec theta, int ysum, int clustsize, int min_idx, int inv, const arma::vec & y, const arma::mat & X, int nparams, arma::vec & res, arma::mat & dres){

  arma::vec yi = y.rows(min_idx - 1, min_idx + clustsize - 2);

  if (ysum == 1) {

    arma::mat Xi  = X.rows( min_idx - 1, min_idx + clustsize - 2 );
    arma::vec wi =  exp( Xi * theta );

    double sumwi = sum( wi );
    arma::vec yihat = wi / sumwi;

    arma::mat yiXi = Xi; 
    yiXi.each_col() %= yihat; 

    arma::mat dresi = -Xi ;
    dresi.each_row() += sum(yiXi, 0);
    dresi.each_col() %= yihat;

    res.rows(min_idx - 1, min_idx + clustsize - 2) = yi - yihat;
    dres.rows(min_idx - 1, min_idx + clustsize - 2) = dresi  ;

  } else {

    arma::mat Xi(clustsize, nparams);

    if(inv) {
      Xi  = -X.rows( min_idx - 1, min_idx + clustsize - 2 );
      ysum = clustsize - ysum;
    } else {
      Xi  = X.rows( min_idx - 1, min_idx + clustsize - 2 );
    }

    arma::vec wi =  exp( Xi * theta );

    arma::vec b(ysum + 1, arma::fill::zeros);
    b(0) = 1;
    b(1) = wi(0);

    arma::mat Db(ysum + 1, nparams, arma::fill::zeros);
    Db.row(1) = Xi.row(0) * wi(0);

    for (int j = 1; j < clustsize; j++) {

      // Set element of the first row
      // b(1) = b(1) + wi(j);
      int maxrow = std::min(j + 1, ysum); 

      arma::mat Db_plus_bXi = Db.rows(0, ysum - 1) + b.rows(0, ysum - 1) * Xi.row(j);
      Db.rows(1, ysum) = Db.rows(1, ysum) + Db_plus_bXi * wi(j);
      b.subvec(1, maxrow) = b.subvec(1, maxrow) + b.subvec(0, maxrow - 1) * wi(j);
    }

    arma::mat Pwi(clustsize, ysum + 1, arma::fill::ones);

    for(int l = 1; l < ysum + 1; l++) {
      Pwi.col(l) = Pwi.col(l - 1) % -wi;
    }

    arma::vec cwi = - Pwi.tail_cols(ysum) * flipud(b.rows(0, ysum - 1));
    arma::vec yihat = cwi / b( ysum );

    arma::mat Dc(clustsize, nparams, arma::fill::zeros); 
    arma::mat Dc_sum_add(clustsize, nparams); 

    for (int l = 0; l < ysum; l++) {
      Dc_sum_add = Xi * (ysum - 1 - l) * b(l);
      Dc_sum_add.each_row() += Db.row(l);  
      Dc_sum_add.each_col() %= Pwi.col(ysum - 1 - l);  
      Dc = Dc + Dc_sum_add;
    }

    arma::mat Xiyihat = Xi;
    Xiyihat.each_col() %= yihat;
    arma::mat Dcwi = Dc;
    Dcwi.each_col() %= wi;
    arma::mat yihatDb = yihat * Db.tail_rows(1);

    if(inv) {
      res.rows(min_idx - 1, min_idx + clustsize - 2) = yi + yihat - 1;
      dres.rows(min_idx - 1, min_idx + clustsize - 2) = Xiyihat - ( yihatDb - Dcwi ) / b(ysum);
    } else {
      res.rows(min_idx - 1, min_idx + clustsize - 2) = yi - yihat;
      dres.rows(min_idx - 1, min_idx + clustsize - 2) = -Xiyihat + ( yihatDb - Dcwi ) / b(ysum);
    }
  }
}

// [[Rcpp::export]]
RcppExport SEXP _conditRes(SEXP thetahat, SEXP ysums, SEXP clustsizes, SEXP minidx, SEXP inv, SEXP yin, SEXP Xcent){
  Rcpp::IntegerVector y_sums(ysums);
  Rcpp::IntegerVector clust_sizes(clustsizes);
  Rcpp::IntegerVector min_idx(minidx);
  Rcpp::IntegerVector invert(inv);
  Rcpp::NumericMatrix X_cent(Xcent);
  Rcpp::NumericVector thetatmp(thetahat);
  Rcpp::NumericVector ytmp(yin);

  int ndisc = y_sums.length(), tot_rows = X_cent.nrow(), nparams =  thetatmp.size();
  arma::vec theta = Rcpp::as<arma::vec>(thetatmp);
  arma::vec y = Rcpp::as<arma::vec>(ytmp);
  arma::vec res(tot_rows, arma::fill::zeros);
  arma::mat dres(tot_rows, nparams, arma::fill::zeros);
  arma::mat X = Rcpp::as<arma::mat>(X_cent);

  for(int j=0; j < ndisc; j++){
    conditResClust(theta, y_sums[j], clust_sizes[j], min_idx[j], invert[j], y, X, nparams, res, dres);
  }

  return Rcpp::List::create( Rcpp::Named("res") = res,
  		       Rcpp::Named("dres") = dres
  		      );
}

// [[Rcpp::export]]
RcppExport SEXP _center(SEXP Uin, SEXP ID) {
  Rcpp::NumericMatrix U_tmp(Uin);
  arma::mat U = Rcpp::as<arma::mat>(U_tmp);

  Rcpp::IntegerVector id_tmp(ID);
  arma::uvec id = Rcpp::as<arma::uvec>(id_tmp);
  arma::uvec uid = unique(id);

  int n_obs = U.n_rows;
  int n_col = U.n_cols;
  // int n_clust = uid.n_elem;
  // int n_clust(nclust);

  // mat Uc_means(size(U));
  arma::mat Uout(U);

  // Initialize the cluster identifier flag
  unsigned int clust_id = id[0];
  // Initialize the cluster index flags
  unsigned int clust_start_idx = 0;
  unsigned int clust_end_idx = 0;

  // double c_sum = 0;
  // Temporary vector to store the sums
  // for each column in the matrix Uin
  // for each cluster
  arma::vec u_sums(n_col);
  u_sums.fill(0.0);
  // int c_size = 0; 

  // Loop over the rows of the matrix Uin
  for(int k = 0; k < n_obs; ++k){

    // If id[k] is a new cluster
    if( id[k] != clust_id ){

      // Update the out matrix
      clust_end_idx = k - 1;

      // Loop over the columns of U for the cluster id[k]
      // to create centered elements in Uout
      for(int l = 0; l < n_col; ++l){
	Uout(arma::span(clust_start_idx, clust_end_idx),l) -= u_sums(l) / (clust_end_idx - clust_start_idx + 1);
	// Uout(clust_start_idx,l) = u_sums(l) / (clust_end_idx - clust_start_idx + 1);
	// Uout(clust_end_idx,l) = u_sums(l) / (clust_end_idx - clust_start_idx + 1);
	// Uout(clust_start_idx,l) = u_sums(l) ;
	// Uout(clust_end_idx,l) = u_sums(l) ;
	// Uout(clust_end_idx,l) = k;
      }

      // Update the cluster_id to the new one
      clust_id = id[k];
      // Update the cluster_start_idx to the new one
      clust_start_idx = k;
      // Update the cluster sums
      u_sums = U.row(k).t();
      // // Reset the cluster sums
      // u_sums.fill(0.0);

    } else {

      // Update the cluster sums
      u_sums += U.row(k).t();

    }

  }

  // Center the last cluster
  clust_end_idx = n_obs - 1;

  for(int l = 0; l < n_col; ++l){
    Uout(arma::span(clust_start_idx, clust_end_idx),l) -= u_sums(l) / (clust_end_idx - clust_start_idx + 1);
    // Uout(clust_start_idx,l) -= u_sums(l) / (clust_end_idx - clust_start_idx + 1);
    // Uout(clust_end_idx,l) -= u_sums(l);

  }

  // Uout.rows(clust_start_idx, clust_end_idx).each_col() -= u_sums / (clust_end_idx - clust_start_idx + 1);

  return Rcpp::wrap(Uout);

}

static const R_CallMethodDef CallEntries[] = {
    {"center", (DL_FUNC) &_center, 2},
    {"conditRes", (DL_FUNC) &_conditRes, 7},
    {NULL, NULL, 0}
};
 
// Register routines and disable symbol search
RcppExport void R_init_drgee(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

