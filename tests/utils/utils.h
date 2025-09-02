#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include "../../include/hmat_lib/hodlr.h"

#define DELTA 1e-10f

void expect_leaf_offdiagonal(struct HODLRLeafNode *leaf,
                            struct HODLRInternalNode *parent);

void expect_leaf_diagonal(struct HODLRLeafNode *leaf,
                          struct HODLRInternalNode *parent);
 
void expect_internal(struct HODLRInternalNode *node,
                     struct HODLRInternalNode *parent);
 
 
int expect_tree_consistent(struct TreeHODLR *hodlr, 
                           int height,
                           const long max_depth_n);

int expect_off_diagonal(
  const struct NodeOffDiagonal *actual,
  const struct NodeOffDiagonal *expected,
  const char *buffer
);

void expect_off_diagonal_decompress(
  const struct NodeOffDiagonal *actual,
  const struct NodeOffDiagonal *expected,
  const int ld_expected,
  const char *buffer,
  double *workspace,
  double *workspace2,
  double *norm_out,
  double *diff_out,
  const double delta
);

int expect_tree_hodlr(struct TreeHODLR *actual, struct TreeHODLR *expected);

void expect_hodlr_decompress(
  const bool fake_hodlr,
  const struct TreeHODLR *actual, 
  const struct TreeHODLR *expected,
  double *workspace,
  double *workspace2,
  double *norm_out,
  double *diff_out,
  const double delta
  );
 
void log_matrix(const double *matrix, const int m, const int n, const int lda);

void expect_matrix_double_eq(const double *actual, 
                             const double *expected, 
                             const int m, 
                             const int n,
                             const int ld_actual,
                             const int ld_expected,
                             const char name,
                             double *norm_out,
                             double *diff_out);

void expect_matrix_double_eq_custom(
    const double *actual, 
    const double *expected, 
    const int m, 
    const int n,
    const int ld_actual,
    const int ld_expected,
    const char name,
  double *norm_out,
  double *diff_out,
    const double delta
);

int expect_matrix_double_eq_safe(
  const double *actual, 
  const double *expected, 
  const int m_actual, 
  const int n_actual,
  const int m_expected, 
  const int n_expected,
  const int ld_actual, 
  const int ld_expected,
  const char name,
  const char *metadata,
  double *norm_out,
  double *diff_out
);

int expect_matrix_double_eq_custom_safe(
  const double *actual, 
  const double *expected, 
  const int m_actual, 
  const int n_actual,
  const int m_expected, 
  const int n_expected,
  const int ld_actual, 
  const int ld_expected,
  const char name,
  const char *metadata,
  double *norm_out,
  double *diff_out,
  const double delta
);


int expect_vector_double_eq_safe(
  const double *actual,
  const double *expected,
  const int len_actual,
  const int len_expected,
  const char name,
  double *norm_out,
  double *diff_out
);

int expect_vector_double_eq_custom(
  const double *actual,
  const double *expected,
  const int len_actual,
  const int len_expected,
  const char name,
  double *norm_out,
  double *diff_out,
  const double delta
);


void print_matrix(int m, int n, double *matrix, int lda);

