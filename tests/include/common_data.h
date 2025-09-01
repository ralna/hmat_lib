#include "../../include/hodlr.h"

double * construct_any_matrix(const int m, 
                              void(*matrix_func)(const int, double *));

double * construct_laplacian_matrix(int m);
void fill_laplacian_matrix(const int m, double *matrix);
void fill_laplacian_converse_matrix(const int m, double *matrix);

double * construct_identity_matrix(int m);
void fill_identity_matrix(const int m, double *matrix);

double * construct_full_matrix(const int m, const double val);
void fill_full_matrix(const int m, const double val, double *matrix);

double * construct_random_matrix(const int m, const int n);
void fill_random_matrix(const int m, const int n, double *matrix);

void fill_decay_matrix(
  const int m, 
  const double *vec, 
  const double scaling_factor,
  double *matrix
);

void fill_decay_matrix_random(
  const int m,
  const double scaling_factor,
  double *matrix
);

void fill_decay_matrix_random_sorted(
  const int m,
  const double scaling_factor,
  double *matrix
);

void construct_fake_hodlr(
  struct TreeHODLR *hodlr, double *matrix, const int s, const int *ss
);
 
void fill_tridiag_symmetric1_matrix(const int m, double *matrix);

