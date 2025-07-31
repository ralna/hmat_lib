#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include <math.h>
#include <time.h>
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

  int m; clock_t start, end;
  start = clock();
  double *matrix = read_dense_matrix("data/H", &m, &cr_malloc, &cr_free);
  end = clock();
  printf("Matrix read in %f s\n", ((double) (end - start)) / CLOCKS_PER_SEC);

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

  clock_t start, end;
  start = clock();

  // Copy matrix
  double *matrix = malloc(matrix_size);
  //memcpy(matrix, params->matrix, matrix_size);
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      matrix[i + j * m] = params->matrix[i + j * m];
    }
  }
  end = clock();
  cr_log_info("Matrix copied! (in %f s)", 
              ((double) (end - start)) / CLOCKS_PER_SEC);

  // Set up HODLR
  struct TreeHODLR *hodlr = allocate_tree_monolithic(params->height, &ierr,
                                                     &malloc, &free);
  cr_log_info("HODLR allocated");
  expect_tree_consistent(hodlr, params->height, hodlr->len_work_queue);

  start = clock();
  dense_to_tree_hodlr(hodlr, params->m, params->ms, matrix, 
                      svd_threshold, &ierr, &malloc, &free);
  free(matrix); matrix = NULL;
  end = clock();
  cr_log_info("HODLR constructed! (in %f s)",
              ((double) (end - start)) / CLOCKS_PER_SEC);

  // Set up vector
  srand(42);
  double *vector = malloc(m * sizeof(double));
  fill_random_matrix(m, 1, vector);
  
  // Reference vector mult
  double *vector_expected = malloc(m * sizeof(double));
  start = clock();
  dgemv_("N", &m, &m, &alpha, params->matrix, &m, vector, &inc, 
         &beta, vector_expected, &inc);
  end = clock();
  cr_log_info("Expected vector computed! (in %f s)",
              ((double) (end - start)) / CLOCKS_PER_SEC);

  // HODLR x vector
  start = clock();
  double *vector_actual = multiply_vector(hodlr, vector, NULL);
  end = clock();
  cr_log_info("Vector multiplied! (in %f s)",
              ((double) (end - start)) / CLOCKS_PER_SEC);

  // Check vector operation
  double norm, diff;
  expect_vector_double_eq_custom(
    vector_actual, vector_expected, m, m, 'V', &norm, &diff, 1e-6
  );
  free(vector_expected); free(vector_actual); free(vector);
  cr_log_info("normv=%f, diff=%f, relerr=%f", sqrtf(norm), sqrtf(diff),
              sqrtf(diff) / sqrtf(norm));

  // Reference matrix mult
  double *matrix_expected = malloc(matrix_size);
  start = clock();
  dgemm_("N", "N", &m, &m, &m, &alpha, params->matrix, &m, params->matrix, &m,
         &beta, matrix_expected, &m);
  end = clock();
  cr_log_info("Expected matrix computed! (in %f s)",
              ((double) (end - start)) / CLOCKS_PER_SEC);

  // HODLR x dense matrix
  start = clock();
  double *matrix_actual = multiply_hodlr_dense(hodlr, params->matrix, 
                                               m, m, NULL, m);
  end = clock();
  cr_log_info("Matrix multiplied! (in %f s)",
              ((double) (end - start)) / CLOCKS_PER_SEC);

  // Check matrix operation
  expect_matrix_double_eq_custom(matrix_actual, matrix_expected, m, m, m, m, 
                                 'M', &norm, &diff, 1e-6);
  cr_log_info("normv=%f, diff=%f, relerr=%f", sqrtf(norm), sqrtf(diff),
              sqrtf(diff) / sqrtf(norm));

  free(matrix_expected); free(matrix_actual);
}

