#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../../include/tree.h"
#include "../../include/blas_wrapper.h"

#include "../include/io.h"
#include "../include/utils.h"
#include "../include/common_data.h"


struct Parameters {
  int height;
  double *matrix;
  int m;
  int *ms;
};


void free_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct Parameters *param = (struct Parameters *) params->params + i;

    cr_free(param->ms);
  }
  struct Parameters *param = (struct Parameters *) params->params;
  cr_free(param->matrix);
  cr_free(params->params);
}


ParameterizedTestParameters(real_data, H) {
  const int n_params = 1;
  struct Parameters *params = cr_malloc(n_params * sizeof(struct Parameters));

  int m, ierr;
  double *matrix = read_dense_matrix("data/H", &m, &cr_malloc, &cr_free);

  const int heights[] = {4};
  const int *ms[] = {NULL};

  for (int i = 0; i < n_params; i++) {
    params[i].matrix = matrix;
    params[i].m = m;
    params[i].height = heights[i];
    params[i].ms = ms[i];
  }

  return cr_make_param_array(struct Parameters, params, n_params, free_params);
}


ParameterizedTest(struct Parameters *params, real_data, H) {
  cr_log_info("height=%d, ms=%p", params->height, params->ms);

  const size_t matrix_size = params->m * params->m * sizeof(double);
  const double svd_threshold = 1e-8, alpha = 1.0, beta = 0.0;
  int ierr = 0; const int m = params->m, inc = 1;

  double *matrix = malloc(matrix_size);
  //memcpy(matrix, params->matrix, matrix_size);
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      matrix[i + j * m] = params->matrix[i + j * m];
    }
  }
  cr_log_info("Matrix copied!");

  struct TreeHODLR *hodlr = allocate_tree_monolithic(params->height, &ierr,
                                                     &malloc, &free);
  cr_log_info("HODLR allocated");
  expect_tree_consistent(hodlr, params->height, hodlr->len_work_queue);
  dense_to_tree_hodlr(hodlr, params->m, params->ms, matrix, 
                      svd_threshold, &ierr, &malloc, &free);
  free(matrix); matrix = NULL;
  cr_log_info("HODLR constructed!");
  expect_tree_consistent(hodlr, params->height, hodlr->len_work_queue);

  srand(42);
  double *vector = construct_random_matrix(params->m, 1);
  
  double *vector_expected = malloc(m * sizeof(double));
  dgemv_("N", &m, &m, &alpha, params->matrix, &m, vector, &inc, 
         &beta, vector_expected, &inc);

  double *vector_actual = multiply_vector(hodlr, vector, NULL);

  double norm, diff;
  expect_vector_double_eq_safe(vector_actual, vector_expected, m, m, 'V',
                               &norm, &diff);
  free(vector_expected); free(vector_actual); free(vector);
  cr_log_info("normv=%f, diff=%f, relerr=%f", sqrtf(norm), sqrtf(diff),
              sqrtf(diff) / sqrtf(norm));

  double *matrix_expected = malloc(matrix_size);
  dgemm_("N", "N", &m, &m, &m, &alpha, params->matrix, &m, params->matrix, &m,
         &beta, matrix_expected, &m);

  double *matrix_actual = multiply_hodlr_dense(hodlr, params->matrix, 
                                               m, m, NULL, m);

  expect_matrix_double_eq(matrix_actual, matrix_expected, m, m, m, m, 'M');
  free(matrix_expected); free(matrix_actual);
}
