#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../include/utils.h"
#include "../include/common_data.h"
#include "../include/tree_stubs.h"

#include "../../src/tree.c"


struct ParametersTestHxV {
  struct TreeHODLR *hodlr;
  double *vector;
  double *expected;
  int len;
};


void free_hv_params(struct criterion_test_params *params) {
  for (int i = 0; i < params->length; i++) {
    struct ParametersTestHxV *param = (struct ParametersTestHxV *) params->params + i;
    
    free_tree_hodlr_cr(&(param->hodlr));
    cr_free(param->expected);
    cr_free(param->vector);
    //cr_free(param);
  }
  cr_free(params->params);
}


struct ParametersTestHxV * generate_compress_params(int * len) {
  int n_params = 2, ierr = 0;
  *len = n_params;
  struct ParametersTestHxV *params = cr_malloc(n_params * sizeof(struct ParametersTestHxV));
  double svd_threshold = 0.1;
  params[0].len = 21;
  params[0].hodlr = allocate_tree_cr(1, &ierr);
  dense_to_tree_hodlr_cr(params[0].hodlr, params[0].len, 
                         construct_laplacian_matrix(params[0].len), svd_threshold, &ierr);

  params[1].len = 21;
  params[1].hodlr = allocate_tree_cr(1, &ierr);

  double *matrix = construct_laplacian_matrix(params[1].len);
  matrix[params[1].len - 1] = 0.5;
  matrix[params[1].len * (params[1].len - 1)] = 0.5;
  dense_to_tree_hodlr_cr(params[1].hodlr, params[1].len,
                         matrix, svd_threshold, &ierr);
  cr_free(matrix);

  for (int j = 0; j < 2; j++) {
    params[j].vector = cr_malloc(params[j].len * sizeof(double));
    for (int i = 0; i < params[j].len; i++) {
      params[j].vector[i] = 10;
    }
    params[j].expected = cr_calloc(params[j].len, sizeof(double));
  }

  params[0].expected[0] = 10;
  params[0].expected[params[0].len - 1] = 10;

  params[1].expected[0] = 15;
  params[1].expected[params[1].len - 1] = 15;
  
  return params;
}


ParameterizedTestParameters(tree, test_hodlr_vector) {
  int n_params;
  struct ParametersTestHxV *params = generate_compress_params(&n_params);

  return cr_make_param_array(struct ParametersTestHxV, params, n_params, free_hv_params);
}


ParameterizedTest(struct ParametersTestHxV *params, tree, test_hodlr_vector) {
  int ierr = 0;

  double * result = multiply_vector(params->hodlr, params->vector, NULL);

  expect_vector_double_eq_safe(result, params->expected, params->len, params->len, 'V');
}

