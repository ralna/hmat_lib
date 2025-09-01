#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include <math.h>
#include <stdio.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../include/utils.h"
#include "../include/common_data.h"

#include "../../src/vector_algebra.c"


#define STR_LEN 10


static inline void fill_full_vector(const int len, const double val, 
                                    double *vector) {
  for (int i = 0; i < len; i++) {
    vector[i] = val;
  }
}


struct ParametersTestHxV {
  struct TreeHODLR *hodlr;
  double *vector;
  double *expected;
  int len;
  char hodlr_name[STR_LEN];
  char vector_name[STR_LEN];
};


void free_hv_params(struct criterion_test_params *params) {
  for (int i = 0; i < params->length; i++) {
    struct ParametersTestHxV *param = (struct ParametersTestHxV *) params->params + i;
    
    free_tree_hodlr(&(param->hodlr), &cr_free);
    cr_free(param->expected);
    cr_free(param->vector);
  }
  cr_free(params->params);
}


int laplacian_matrix(struct ParametersTestHxV *params) {
  int i = 0, ierr = 0;
  const int n_cases = 3, len = 21, max_height = 3;
  double svd_threshold = 0.1;
  double *matrix = cr_malloc(len * len * sizeof(double));

  for (int height = 1; height < max_height+1; height++) {
    i = n_cases * (height - 1);

    for (int j = i; j < i+n_cases; j++) {
      params[j].len = len;
      params[j].hodlr = allocate_tree_monolithic(height, &ierr, &cr_malloc, &cr_free);
    }

    strncat(params[i].hodlr_name, "L", STR_LEN);
    fill_laplacian_matrix(len, matrix);
    dense_to_tree_hodlr(params[i].hodlr, params[i].len, NULL, 
                        matrix, svd_threshold, &ierr, &cr_malloc, &cr_free);

    strncat(params[i+1].hodlr_name, "L0.5S", STR_LEN);
    fill_laplacian_matrix(len, matrix);
    matrix[params[i+1].len - 1] = 0.5;
    matrix[params[i+1].len * (params[i+1].len - 1)] = 0.5;
    dense_to_tree_hodlr(params[i+1].hodlr, params[i+1].len, NULL,
                        matrix, svd_threshold, &ierr, &cr_malloc, &cr_free);

    strncat(params[i+2].hodlr_name, "L0.5A", STR_LEN);
    fill_laplacian_matrix(len, matrix);
    matrix[params[i+2].len - 1] = 0.5;
    dense_to_tree_hodlr(params[i+2].hodlr, params[i+2].len, NULL,
                        matrix, svd_threshold, &ierr, &cr_malloc, &cr_free);

    for (int j = 0; j < n_cases; j++) {
      strncat(params[i+j].vector_name, "10", STR_LEN);
      params[i+j].vector = cr_malloc(params[i+j].len * sizeof(double));
      for (int k = 0; k < params[i+j].len; k++) {
        params[i+j].vector[k] = 10.;
      }
      params[i+j].expected = cr_calloc(params[i+j].len, sizeof(double));
    }

    params[i].expected[0] = 10.;
    params[i].expected[params[i].len - 1] = 10.;

    params[i+1].expected[0] = 15.;
    params[i+1].expected[params[i+1].len - 1] = 15.;

    params[i+2].expected[0] = 10.;
    params[i+2].expected[params[i+2].len - 1] = 15.;
  }
  cr_free(matrix);

  return max_height * n_cases;
}


int identity_matrix(struct ParametersTestHxV *params) {
  int i = 0, idx = 0, ierr = 0;
  const int n_cases = 3, len = 21, max_height = 3;
  double svd_threshold = 0.1;
  double *matrix = cr_malloc(len * len * sizeof(double));

  for (int height = 1; height < max_height+1; height++) {
    for (i = idx; i < idx+n_cases; i++) {
      params[i].len = len;
      params[i].hodlr = allocate_tree_monolithic(height, &ierr, 
                                                 &cr_malloc, &cr_free);

      strncat(params[i].hodlr_name, "I", STR_LEN);
      fill_identity_matrix(len, matrix);
      dense_to_tree_hodlr(params[i].hodlr, len, NULL,
                          matrix, svd_threshold, &ierr, &cr_malloc, &cr_free);
      params[i].vector = cr_calloc(params[i].len, sizeof(double));
      params[i].expected = cr_calloc(params[i].len, sizeof(double));
    }

    strncat(params[idx].vector_name, "10", STR_LEN);
    for (int k = 0; k < params[idx].len; k++) {
      params[idx].vector[k] = 10.;
      params[idx].expected[k] = 10.;
    }
    idx++;

    strncat(params[idx].vector_name, "0", STR_LEN);
    for (int k = 0; k < params[idx+1].len; k++) {
      params[idx].vector[k] = 0.;
      params[idx].expected[k] = 0.;
    }
    idx++;

    strncat(params[idx].vector_name, "k..-10", STR_LEN);
    for (int k = 0; k < params[idx+2].len; k++) {
      params[idx].vector[k] = (double)(k - 10);
      params[idx].expected[k] = (double)(k - 10);
    }
    idx++;
  }

  cr_free(matrix);
  return max_height * n_cases;
}


struct ParametersTestHxV * generate_hodlr_vector_params(int * len) {
  const int n_params = 9+9;
  *len = n_params;
  struct ParametersTestHxV *params = 
    cr_malloc(n_params * sizeof(struct ParametersTestHxV));

  int actual = laplacian_matrix(params);
  actual += identity_matrix(params + actual);

  if (actual != n_params) {
    printf("PARAMETER SET-UP FAILED - allocated %d parameters but set %d\n",
           n_params, actual);
  }

  return params;
}


ParameterizedTestParameters(vector_algebra, test_hodlr_vector) {
  int n_params;
  struct ParametersTestHxV *params = generate_hodlr_vector_params(&n_params);

  return cr_make_param_array(struct ParametersTestHxV, params, n_params, 
                             free_hv_params);
}


ParameterizedTest(struct ParametersTestHxV *params, vector_algebra, 
                  test_hodlr_vector) {
  cr_log_info("hodlr '%s' (height=%d, m=%d) x vector '%s' (l=%d)",
              params->hodlr_name, params->hodlr->height, 
              params->hodlr->root->m, params->vector_name, params->len);

  int ierr;
  double * result = 
    multiply_vector(params->hodlr, params->vector, NULL, &ierr);

  cr_expect(eq(int, ierr, SUCCESS));

  double norm, diff;
  expect_vector_double_eq_safe(result, params->expected, params->len, 
                               params->len, 'V', &norm, &diff);
  cr_log_info("normv=%f, diff=%f, relerr=%f", sqrtf(norm), sqrtf(diff),
              sqrtf(diff) / sqrtf(norm));

  free(result);
}


struct ParametersTestOffdiagxV {
  struct HODLRInternalNode *parent;
  double *vector;
  double *out;
  double *expected;
  int len;
  int offset;
  char hodlr_name[STR_LEN];
  char vector_name[STR_LEN];
  char out_name[STR_LEN];
};


void free_ov_params(struct criterion_test_params *params) {
  for (int i = 0; i < params->length; i++) {
    struct ParametersTestOffdiagxV *param = 
      (struct ParametersTestOffdiagxV *) params->params + i;

    for (int i = 1; i < 3; i++) {
      cr_free(param->parent->children[i].leaf->data.off_diagonal.u);
      cr_free(param->parent->children[i].leaf->data.off_diagonal.v);
      cr_free(param->parent->children[i].leaf);
    }
    cr_free(param->parent);

    cr_free(param->expected);
    cr_free(param->out);
    cr_free(param->vector);
  }
  cr_free(params->params);
}


ParameterizedTestParameters(vector_algebra, multiply_off_diagonal_vector) {
  const int n_params = 3;
  struct ParametersTestOffdiagxV *params = 
    cr_malloc(n_params * sizeof(struct ParametersTestOffdiagxV));

  const int lens[] = {20, 10, 19};
  const int ms[] = {6, 4, 8};
  const int ns[] = {5, 4, 7};
  const int ss[] = {1, 1, 1};
  const int offsets[] = {5, 2, 3};

  for (int i = 0; i < n_params; i++) {
    params[i].parent = cr_malloc(sizeof(struct HODLRInternalNode));
    params[i].parent->children[1].leaf = 
      cr_malloc(sizeof(struct NodeOffDiagonal));
    params[i].parent->children[1].leaf->data.off_diagonal.u =
      cr_calloc(ms[i] * ss[i], sizeof(double));
    params[i].parent->children[1].leaf->data.off_diagonal.v =
      cr_calloc(ns[i] * ss[i], sizeof(double));
    params[i].parent->children[1].leaf->data.off_diagonal.m = ms[i];
    params[i].parent->children[1].leaf->data.off_diagonal.n = ns[i];
    params[i].parent->children[1].leaf->data.off_diagonal.s = ss[i];

    params[i].parent->children[2].leaf = 
      cr_malloc(sizeof(struct NodeOffDiagonal));
    params[i].parent->children[2].leaf->data.off_diagonal.m = ns[i];
    params[i].parent->children[2].leaf->data.off_diagonal.n = ms[i];
    params[i].parent->children[2].leaf->data.off_diagonal.s = ss[i];
    params[i].parent->children[2].leaf->data.off_diagonal.u =
      cr_calloc(ns[i] * ss[i], sizeof(double));
    params[i].parent->children[2].leaf->data.off_diagonal.v =
      cr_calloc(ms[i] * ss[i], sizeof(double));
    
    params[i].offset = offsets[i];
    params[i].len = lens[i];
    params[i].vector = cr_calloc(lens[i], sizeof(double));
    params[i].out = cr_calloc(lens[i], sizeof(double));
    params[i].expected = cr_calloc(lens[i], sizeof(double));
  }

  // 0 HODLR x 10 vector
  int i = 0;
  strncat(params[i].hodlr_name, "0", STR_LEN);
  strncat(params[i].vector_name, "10", STR_LEN);
  fill_full_vector(lens[i], 10., params[i].vector);
  strncat(params[i].out_name, "0", STR_LEN);
  fill_full_vector(lens[i], 0., params[i].out);
  fill_full_vector(lens[i], 0., params[i].expected);

  // 5 HODLR x 0.5 vector
  i++;
  strncat(params[i].hodlr_name, "+/-5", STR_LEN);
  fill_full_vector(ms[i] * ss[i], 5., 
                   params[i].parent->children[1].leaf->data.off_diagonal.u);
  fill_full_vector(ns[i] * ss[i], 1., 
                   params[i].parent->children[1].leaf->data.off_diagonal.v);
  fill_full_vector(ns[i] * ss[i], -5., 
                   params[i].parent->children[2].leaf->data.off_diagonal.u);
  fill_full_vector(ms[i] * ss[i], 1., 
                   params[i].parent->children[2].leaf->data.off_diagonal.v);
  strncat(params[i].vector_name, "0.5", STR_LEN);
  fill_full_vector(lens[i], 0.5, params[i].vector);
  strncat(params[i].out_name, "-10", STR_LEN);
  fill_full_vector(lens[i], -10., params[i].out);
  fill_full_vector(lens[i], -10., params[i].expected);
  fill_full_vector(ms[i], 0., params[i].expected + params[i].offset);
  fill_full_vector(ns[i], -20., 
                   params[i].expected + params[i].offset + ms[i]);

  // More realistic
  i++;
  strncat(params[i].hodlr_name, "L", STR_LEN);
  params[i].parent->children[1].leaf->data.off_diagonal.u[ms[i]-1] = 1.;
  params[i].parent->children[1].leaf->data.off_diagonal.v[0] = 1.;
  params[i].parent->children[2].leaf->data.off_diagonal.u[0] = 1.;
  params[i].parent->children[2].leaf->data.off_diagonal.v[ms[i]-1] = 1.;
  strncat(params[i].vector_name, "3.14", STR_LEN);
  fill_full_vector(lens[i], 3.14, params[i].vector);
  strncat(params[i].out_name, "0..len", STR_LEN);
  for (int j = 0; j < lens[i]; j++) {
    params[i].out[j] = (double)j;
    params[i].expected[j] = (double)j;
  }
  params[i].expected[offsets[i] + ms[i] - 1] += 3.14;
  params[i].expected[offsets[i] + ms[i]] += 3.14;

  if (i+1 != n_params) {
    printf("PARAMETER SET-UP FAILED - allocated %d parameters but set %d\n",
           n_params, i+1);
  }

  return cr_make_param_array(struct ParametersTestOffdiagxV, params, n_params, 
                             free_ov_params);
}


ParameterizedTest(struct ParametersTestOffdiagxV *params, vector_algebra, 
                  multiply_off_diagonal_vector) {
  const int m = params->parent->children[1].leaf->data.off_diagonal.m;
  const int n = params->parent->children[1].leaf->data.off_diagonal.n;
  const int s = params->parent->children[1].leaf->data.off_diagonal.s;

  cr_log_info("off-diag '%s' (m=%d, n=%d, s=%d) x vector '%s' (l=%d) + '%s'",
              params->hodlr_name, m, n, s, params->vector_name, params->len,
              params->out_name);

  double *workspace = malloc(m * sizeof(double));

  multiply_off_diagonal_vector(
    params->parent, params->vector + params->offset, 
    params->out + params->offset, workspace
  );

  double norm, diff;
  expect_vector_double_eq_safe(params->out, params->expected, params->len, 
                               params->len, 'V', &norm, &diff);
  cr_log_info("normv=%f, diff=%f, relerr=%f", sqrtf(norm), sqrtf(diff),
              sqrtf(diff) / sqrtf(norm));


  free(workspace);
}


