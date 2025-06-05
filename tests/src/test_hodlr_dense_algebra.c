#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include <string.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../include/utils.h"
#include "../include/common_data.h"

#include "../../include/tree.h"
#include "../../src/dense_algebra.c"


#define STR_LEN 10


static inline void arrset(int *arr, const int len, const int val) {
  for (int i = 0; i < len; i++) {
    arr[i] = val;
  }
}


struct ParametersWorkspaceSize {
  struct TreeHODLR *hodlr;
  int matrix_a;
  int *expected;
  int expected_s;
};


void free_workspace_size_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersWorkspaceSize *param = 
      (struct ParametersWorkspaceSize *) params->params + i;
    
    free_tree_hodlr(&(param->hodlr), &cr_free);
    cr_free(param->expected);
  }
  cr_free(params->params);
}


ParameterizedTestParameters(dense_algebra, compute_workspace_size) {
  const int max_height = 4;
  const size_t ss_len = (size_t)(pow(2, max_height + 1)) - 2;
  const size_t ss_size = ss_len * sizeof(int);
  int *ss = cr_malloc(ss_size);
  
  const int n_params = 2 * max_height;

  struct ParametersWorkspaceSize *params = 
    cr_malloc(n_params * sizeof(struct ParametersWorkspaceSize));
  
  int idx = 0, ierr = SUCCESS, m = 0;
  for (int height = 1; height < max_height+1; height++) {
    arrset(ss, ss_len, height);
    params[idx].expected_s = height;

    params[idx].matrix_a = 2048;
    params[idx].hodlr = allocate_tree_monolithic(height, &ierr, &cr_malloc, &cr_free);
    fill_leaf_node_ints(params[idx].hodlr, 42, ss);

    params[idx].expected = cr_malloc(2 * sizeof(int));
    params[idx].expected[0] = ss[0] * params[idx].matrix_a;
    m = params[idx].hodlr->root->m;
    params[idx].expected[1] = (m - m / 2) * params[idx].matrix_a;
    idx++;
  }

  int highest_s = 0;
  arrset(ss, ss_len, 1);
  ss[0] = max_height + 1;
  for (int height = 1; height < max_height+1; height++) {
    highest_s = max_height + height;
    params[idx].expected_s = highest_s;

    ss[2 * height] = highest_s;
    params[idx].matrix_a = 42;
    params[idx].hodlr = allocate_tree_monolithic(height, &ierr, &cr_malloc, &cr_free);
    fill_leaf_node_ints(params[idx].hodlr, 2048, ss);

    params[idx].expected = cr_malloc(2 * sizeof(int));
    params[idx].expected[0] = highest_s * params[idx].matrix_a;
    m = params[idx].hodlr->root->m;
    params[idx].expected[1] = (m - m / 2) * params[idx].matrix_a;
    idx++;
  }

  if (idx != n_params) {
    printf("INCORRECT PARAMATER SETUP: allocated %d parameters but constructed %d\n",
           n_params, idx);
  }

  cr_free(ss);

  return cr_make_param_array(struct ParametersWorkspaceSize, params, n_params,
                             free_workspace_size_params);
}


ParameterizedTest(struct ParametersWorkspaceSize *params, dense_algebra, 
                  compute_workspace_size) {
  cr_log_info("height=%d, s=%d", params->hodlr->height, params->expected_s);

  int *result = malloc(2 * sizeof(int));
  compute_multiply_hodlr_dense_workspace(params->hodlr, params->matrix_a, result);

  cr_expect(eq(int, result[0], params->expected[0]));
  cr_expect(eq(int, result[1], params->expected[1]));

  free(result);
}


inline static void fill_matrix_row(double *matrix, 
                            const int row,
                            const int lda,
                            const int n_cols,
                            const double val) {
  for (int j = 0; j < n_cols; j++) {
    matrix[row + j * lda] = val;
  }
}


struct ParametersTestHxD {
  struct TreeHODLR *hodlr;
  double *dense;
  double *expected;
  int m;
  int dense_n;
  int dense_ld;
  char hodlr_name[STR_LEN];
  char dense_name[STR_LEN];
};


void free_hd_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersTestHxD *param = (struct ParametersTestHxD *) params->params + i;
    
    free_tree_hodlr(&(param->hodlr), &cr_free);
    cr_free(param->expected);
    cr_free(param->dense);
    //cr_free(param);
  }
  cr_free(params->params);
}


static void laplacian_matrix(struct ParametersTestHxD *params,
                      int start) {
  int i = 0, ierr = 0, n_cases = 3;
  double svd_threshold = 0.1;

  for (int height = 1; height < 4; height++) {
    i = n_cases * (height - 1) + start;

    for (int j = i; j < i+n_cases; j++) {
      params[j].m = 21;
      params[j].dense_n = 21;
      params[j].dense_ld = 21;
      params[j].hodlr = allocate_tree_monolithic(height, &ierr, 
                                                 &cr_malloc, &cr_free);
    }

    // LAPLACIAN MATRIX
    strncat(params[i].hodlr_name, "L", STR_LEN);
    dense_to_tree_hodlr(params[i].hodlr, params[i].m, 
                        construct_laplacian_matrix(params[i].m), 
                        svd_threshold, &ierr, &cr_malloc, &cr_free);

    // LAPLACIAN MATRIX with 0.5 in corners
    strncat(params[i+1].hodlr_name, "L0.5S", STR_LEN);
    double *matrix = construct_laplacian_matrix(params[i+1].m);
    matrix[params[i+1].m - 1] = 0.5;
    matrix[params[i+1].m * (params[i+1].m - 1)] = 0.5;
    dense_to_tree_hodlr(params[i+1].hodlr, params[i+1].m,
                        matrix, svd_threshold, &ierr, &cr_malloc, &cr_free);
    cr_free(matrix);

    // LAPLACIAN MATRIX with 0.5 in bottom corner
    strncat(params[i+2].hodlr_name, "L0.5A", STR_LEN);
    matrix = construct_laplacian_matrix(params[i+2].m);
    matrix[params[i+2].m - 1] = 0.5;
    dense_to_tree_hodlr(params[i+2].hodlr, params[i+2].m,
                        matrix, svd_threshold, &ierr, &cr_malloc, &cr_free);
    cr_free(matrix);

    for (int j = 0; j < n_cases; j++) {
      strncat(params[i+j].dense_name, "10", STR_LEN);
      params[i+j].dense = construct_full_matrix(params[i+j].m, 10.0);
      params[i+j].expected = cr_calloc(params[i+j].m * params[i+j].m, sizeof(double));
    }

    // LAPLACIAN MATRIX
    fill_matrix_row(params[i].expected, 0, params[i].m, params[i].dense_ld, 10.0);
    fill_matrix_row(params[i].expected, params[i].m-1, params[i].m, params[i].dense_ld, 10.0);

    // LAPLACIAN MATRIX with 0.5 in corners
    i += 1;
    fill_matrix_row(params[i].expected, 0, params[i].m, params[i].dense_ld, 15.0);
    fill_matrix_row(params[i].expected, params[i].m-1, params[i].m, params[i].dense_ld, 15.0);

    // LAPLACIAN MATRIX with 0.5 in bottom corner
    i += 1;
    fill_matrix_row(params[i].expected, 0, params[i].m, params[i].dense_ld, 10.0);
    fill_matrix_row(params[i].expected, params[i].m-1, params[i].m, params[i].dense_ld, 15.0);
  }
}


static void identity_matrix(struct ParametersTestHxD *params,
                     int start) {
  int i = 0, idx = 0, ierr = 0, n_cases = 3;
  double svd_threshold = 0.1;

  for (int height = 1; height < 4; height++) {
    idx = n_cases * (height - 1) + start;

    for (i = idx; i < idx+n_cases; i++) {
      params[i].m = 21;
      params[i].dense_n = 21;
      params[i].dense_ld = 21;
      strncat(params[i].hodlr_name, "I", STR_LEN);

      params[i].hodlr = allocate_tree_monolithic(height, &ierr,
                                                 &cr_malloc, &cr_free);
      dense_to_tree_hodlr(params[i].hodlr, params[i].m, 
                          construct_identity_matrix(params[i].m), 
                          svd_threshold, &ierr, &cr_malloc, &cr_free);
    }

    strncat(params[idx].dense_name, "I", STR_LEN);
    params[idx].dense = construct_identity_matrix(params[idx].m);
    params[idx].expected = construct_identity_matrix(params[idx].m);

    strncat(params[idx+1].dense_name, "0", STR_LEN);
    params[idx+1].dense = cr_calloc(params[idx+1].m * params[idx+1].m, sizeof(double));
    params[idx+1].expected = cr_calloc(params[idx+1].m * params[idx+1].m, sizeof(double));

    strncat(params[idx+2].dense_name, "L", STR_LEN);
    params[idx+2].dense = construct_laplacian_matrix(params[idx].m);
    params[idx+2].expected = construct_laplacian_matrix(params[idx].m);
  }
}


struct ParametersTestHxD * generate_hodlr_dense_params(int * len) {
  int n_params = 9+9;
  *len = n_params;
  struct ParametersTestHxD *params = cr_malloc(n_params * sizeof(struct ParametersTestHxD));

  laplacian_matrix(params, 0);
  identity_matrix(params, 9);
  return params;
}


ParameterizedTestParameters(dense_algebra, test_hodlr_dense) {
  int n_params;
  struct ParametersTestHxD *params = generate_hodlr_dense_params(&n_params);

  return cr_make_param_array(struct ParametersTestHxD, params, n_params, free_hd_params);
}


ParameterizedTest(struct ParametersTestHxD *params, dense_algebra, test_hodlr_dense) {
  int ierr = 0;
  int m = params->hodlr->root->m;

  cr_log_info("%.10s (height=%d) x %.10s (%dx%d, lda=%d)",
              params->hodlr_name, params->hodlr->height, params->dense_name, 
              params->m, params->dense_n, params->dense_ld);

  double * result = multiply_hodlr_dense(params->hodlr, params->dense, params->dense_n, 
                                         params->dense_ld, NULL, m);

  expect_matrix_double_eq_safe(result, params->expected, m, params->dense_n, 
                               m, params->dense_n, m, m, 'M');

  free(result);
}

