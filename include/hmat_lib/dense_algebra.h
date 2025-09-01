#pragma once


int compute_multiply_hodlr_dense_workspace(
  const struct TreeHODLR *hodlr,
  const int matrix_a
);


double * multiply_hodlr_dense(
  const struct TreeHODLR *hodlr,
  const double *matrix,
  const int matrix_n,
  const int matrix_ld,
  double *out,
  const int out_ld,
  int *ierr
);


double * multiply_hodlr_transpose_dense(
  const struct TreeHODLR *hodlr,
  const double *matrix,
  const int matrix_n,
  const int matrix_ld,
  double *out,
  const int out_ld,
  int *ierr
);


double * multiply_dense_hodlr(
  const struct TreeHODLR *hodlr,
  const double * matrix,
  const int matrix_m,
  const int matrix_ld,
  double * out,
  const int out_ld,
  int *ierr
);
 

void multiply_internal_node_dense(
  const struct HODLRInternalNode *internal,
  const int height,
  const double *matrix,
  const int matrix_n,
  const int matrix_ld,
  const struct HODLRInternalNode **queue,
  double *workspace,
  double *out,
  const int out_ld
);


void multiply_internal_node_transpose_dense(
  const struct HODLRInternalNode *internal,
  const int height,
  const double *matrix,
  const int matrix_n,
  const int matrix_ld,
  const struct HODLRInternalNode **queue,
  double *workspace,
  double *out,
  const int out_ld
);


void multiply_dense_internal_node(
  const struct HODLRInternalNode *internal,
  const int height,
  const double *matrix,
  const int matrix_m,
  const int matrix_ld,
  const struct HODLRInternalNode **queue,
  double *workspace,
  double *out,
  const int out_ld
);

