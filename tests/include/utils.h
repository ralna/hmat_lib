#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include "../../include/tree.h"

void expect_leaf_offdiagonal(struct HODLRLeafNode *leaf,
                            struct HODLRInternalNode *parent);

void expect_leaf_diagonal(struct HODLRLeafNode *leaf,
                          struct HODLRInternalNode *parent);
 
void expect_internal(struct HODLRInternalNode *node,
                     struct HODLRInternalNode *parent);
 
 
int expect_tree_consistent(struct TreeHODLR *hodlr, 
                           int height,
                           const long max_depth_n);


int expect_tree_hodlr(struct TreeHODLR *actual, struct TreeHODLR *expected);
  
void log_matrix(const double *matrix, const int m, const int n, const int lda);

void expect_matrix_double_eq(const double *actual, 
                             const double *expected, 
                             const int m, 
                             const int n,
                             const int ld_actual,
                             const int ld_expected,
                             const char name);

void expect_matrix_double_eq_custom(
    const double *actual, 
    const double *expected, 
    const int m, 
    const int n,
    const int ld_actual,
    const int ld_expected,
    const char name,
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
  const char *metadata
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

void fill_leaf_node_ints(struct TreeHODLR *hodlr, const int m, int *ss);

