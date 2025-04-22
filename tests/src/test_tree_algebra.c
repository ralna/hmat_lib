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
  int n_params = 6, ierr = 0;
  *len = n_params;
  struct ParametersTestHxV *params = cr_malloc(n_params * sizeof(struct ParametersTestHxV));
  double svd_threshold = 0.1;
  int i = 0;

  for (int height = 1; height < 4; height++) {
    i = 2 * (height - 1);

    params[i].len = 21;
    params[i].hodlr = allocate_tree_cr(height, &ierr);
    dense_to_tree_hodlr_cr(params[i].hodlr, params[i].len, 
                           construct_laplacian_matrix(params[i].len), 
                           svd_threshold, &ierr);

    params[i+1].len = 21;
    params[i+1].hodlr = allocate_tree_cr(height, &ierr);

    double *matrix = construct_laplacian_matrix(params[i+1].len);
    matrix[params[i+1].len - 1] = 0.5;
    matrix[params[i+1].len * (params[i+1].len - 1)] = 0.5;
    dense_to_tree_hodlr_cr(params[i+1].hodlr, params[i+1].len,
                          matrix, svd_threshold, &ierr);
    cr_free(matrix);

    for (int j = 0; j < 2; j++) {
      params[i+j].vector = cr_malloc(params[i+j].len * sizeof(double));
      for (int k = 0; k < params[i+j].len; k++) {
        params[i+j].vector[k] = 10;
      }
      params[i+j].expected = cr_calloc(params[i+j].len, sizeof(double));
    }

    params[i].expected[0] = 10;
    params[i].expected[params[i].len - 1] = 10;

    params[i+1].expected[0] = 15;
    params[i+1].expected[params[i+1].len - 1] = 15;
  }

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

