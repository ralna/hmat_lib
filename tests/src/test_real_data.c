#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../../include/tree.h"
#include "../../include/blas_wrapper.h"

#include "../include/io.h"
#include "../include/utils.h"
#include "../include/common_data.h"

#ifdef HODLR_REAL_DATA_PRINT_S
#include "../../dev/common.h"
#endif


static inline void start_timer(clock_t *cstart, double *wstart) {
  #ifdef _OPENMP
  *wstart = omp_get_wtime();
  #endif
  *cstart = clock();
}


static inline void get_time(clock_t cstart, double wstart, char *str) {
  clock_t cend = clock();
  #ifdef _OPENMP
  double wend = omp_get_wtime();
  cr_log_info("%s (ctime=%f s, wtime=%f s)", str, 
              ((double) (cend - cstart)) / CLOCKS_PER_SEC, 
              wend - wstart);
  #else
  cr_log_info("%s (ctime=%f s)", str, 
              ((double) (cend - cstart)) / CLOCKS_PER_SEC);
  #endif
}


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
  enum {n_params = 5};
  struct Parameters *params = cr_malloc(n_params * sizeof(struct Parameters));

  int m; clock_t start, end;
  start = clock();
  double *matrix = read_dense_matrix("data/H", &m, &cr_malloc, &cr_free);
  end = clock();
  printf("Matrix read in %f s\n", ((double) (end - start)) / CLOCKS_PER_SEC);

  const int heights[n_params] = {1, 2, 3, 4, 5};
  const int *ms[n_params] = {NULL, NULL, NULL, NULL, NULL};

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

  clock_t start; double wstart;

  start_timer(&start, &wstart);

  // Copy matrix
  double *matrix = malloc(matrix_size);
  //memcpy(matrix, params->matrix, matrix_size);
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      matrix[i + j * m] = params->matrix[i + j * m];
    }
  }
  get_time(start, wstart, "Matrix copied!");

  // Set up HODLR
  struct TreeHODLR *hodlr = allocate_tree_monolithic(params->height, &ierr,
                                                     &malloc, &free);
  cr_log_info("HODLR allocated");
  expect_tree_consistent(hodlr, params->height, hodlr->len_work_queue);

  start_timer(&start, &wstart);
  dense_to_tree_hodlr(hodlr, params->m, params->ms, matrix, 
                      svd_threshold, &ierr, &malloc, &free);
  free(matrix); matrix = NULL;
  get_time(start, wstart, "HODLR constructed!");

#ifdef HODLR_REAL_DATA_PRINT_S
  log_hodlr_s_symmetric(hodlr);
#endif

  // Set up vector
  srand(42);
  double *vector = malloc(m * sizeof(double));
  fill_random_matrix(m, 1, vector);
  
  // Reference vector mult
  double *vector_expected = malloc(m * sizeof(double));
  start_timer(&start, &wstart);
  dgemv_("N", &m, &m, &alpha, params->matrix, &m, vector, &inc, 
         &beta, vector_expected, &inc);
  get_time(start, wstart, "Expected vector computed!");

  // HODLR x vector
  start_timer(&start, &wstart);
  double *vector_actual = multiply_vector(hodlr, vector, NULL);
  get_time(start, wstart, "Vector multiplied!");

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
  start_timer(&start, &wstart);
  dgemm_("N", "N", &m, &m, &m, &alpha, params->matrix, &m, params->matrix, &m,
         &beta, matrix_expected, &m);
  get_time(start, wstart, "Expected matrix computed!");

  // HODLR x dense matrix
  start_timer(&start, &wstart);
  double *matrix_actual = multiply_hodlr_dense(hodlr, params->matrix, 
                                               m, m, NULL, m);
  get_time(start, wstart, "Matrix multiplied!");

  // Check matrix operation
  expect_matrix_double_eq_custom(matrix_actual, matrix_expected, m, m, m, m, 
                                 'M', &norm, &diff, 1e-6);
  cr_log_info("normv=%f, diff=%f, relerr=%f", sqrtf(norm), sqrtf(diff),
              sqrtf(diff) / sqrtf(norm));

  free(matrix_expected); free(matrix_actual);
}

