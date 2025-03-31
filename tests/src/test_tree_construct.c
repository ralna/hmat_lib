#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../include/utils.h"
#include "../include/common_data.h"

#include "../../src/tree.c"


struct ParametersTestCompress {
  int m_full;
  int m;
  int n;
  double svd_threshold;
  int expected_n_singular;
  double *matrix;
  double *u_expected;
  double *v_expected;
  double *full_matrix;
};


void free_compress_params(struct criterion_test_params *params) {
  for (int i = 0; i < params->length; i++) {
    struct ParametersTestCompress *param = params->params + i;
    cr_free(param->full_matrix);
    cr_free(param->u_expected);
    cr_free(param->v_expected);
    //cr_free(param);
  }
  cr_free(params->params);
}


struct ParametersTestCompress * generate_compress_params(int * len) {
  int n_params = 4;
  *len = n_params;
  struct ParametersTestCompress *params = cr_malloc(n_params * sizeof(struct ParametersTestCompress));

  for (int i = 0; i < 2; i++) {
    params[i].m_full = 10;
    params[i].m = 5;
    params[i].n = 5;
    params[i].svd_threshold = 0.1;
    params[i].expected_n_singular = 1;
    params[i].full_matrix = construct_laplacian_matrix(params[i].m_full);
    params[i].u_expected = cr_calloc(params[i].m * params[i].expected_n_singular, sizeof(double));
    params[i].v_expected = cr_calloc(params[i].expected_n_singular * params[i].n, sizeof(double));
  }

  params[0].matrix = params[0].full_matrix + params[0].m;
  params[0].u_expected[0] = 1;
  params[0].v_expected[0] = -0;
  params[0].v_expected[4] = -1;

  params[1].matrix = params[1].full_matrix + params[1].m_full * params[1].m;
  params[1].u_expected[4] = 1;
  params[1].v_expected[0] = -1;

  for (int i = 2; i < 4; i++) {
    params[i].m_full = 10;
    params[i].m = 5;
    params[i].n = 5;
    params[i].svd_threshold = 0.1;
    params[i].expected_n_singular = 2;
    params[i].full_matrix = construct_laplacian_matrix(params[i].m_full);
    params[i].full_matrix[params[i].m_full - 1] = 0.5;
    params[i].full_matrix[params[i].m_full * (params[i].m_full - 1)] = 0.5;

    params[i].u_expected = cr_calloc(params[i].m * params[i].expected_n_singular, sizeof(double));
    params[i].v_expected = cr_calloc(params[i].expected_n_singular * params[i].n, sizeof(double));
  }

  params[2].matrix = params[2].full_matrix + params[2].m;
  params[2].u_expected[0] = -1;
  params[2].u_expected[9] = -0.5;

  params[2].v_expected[4] = 1;
  params[2].v_expected[5] = -1;

  params[3].matrix = params[3].full_matrix + params[3].m_full * params[3].m;
  params[3].u_expected[4] = 1;
  params[3].u_expected[5] = 0.5;

  params[3].v_expected[0] = -1;
  params[3].v_expected[9] = 1;

  return params;
}


ParameterizedTestParameters(tree, test_compress) {
  int n_params;
  struct ParametersTestCompress *params = generate_compress_params(&n_params);
  return cr_make_param_array(struct ParametersTestCompress, params, n_params, free_compress_params);
}


ParameterizedTest(struct ParametersTestCompress *params, tree, test_compress) {
  struct NodeOffDiagonal result;
  int n_singular_values = params->m < params->n ? params->m : params->n;
  
  double *s_work = malloc(n_singular_values * sizeof(double));
  double *u_work = malloc(params->m * n_singular_values * sizeof(double));
  double *vt_work = malloc(params->n * n_singular_values * sizeof(double));

  int result_code = compress_off_diagonal(
      &result, params->m, params->n, n_singular_values, params->m_full, 
      params->matrix, s_work, u_work, vt_work, params->svd_threshold
  );

  free(s_work); free(u_work); free(vt_work);

  cr_expect(eq(result_code, 0));
  if (result_code != 0) {
    cr_fatal();
  } 
  cr_expect(eq(result.m, params->m));
  cr_expect(eq(result.s, params->expected_n_singular));
  cr_expect(eq(result.n, params->n));

  expect_arr_double_eq(result.u, params->u_expected, params->m, params->expected_n_singular,
                       result.m, params->m);
  expect_arr_double_eq(result.v, params->v_expected, params->n, params->expected_n_singular,
                       result.m, params->m);
}


ParameterizedTestParameters(tree, test_decompress) {
  int n_params;
  struct ParametersTestCompress *params = generate_compress_params(&n_params);
  return cr_make_param_array(struct ParametersTestCompress, params, n_params, free_compress_params);
}


ParameterizedTest(struct ParametersTestCompress *params, tree, test_decompress) {
  struct NodeOffDiagonal node;
  node.m = params->m;
  node.s = params->expected_n_singular;
  node.n = params->n;

  node.u = params->u_expected;
  node.v = params->v_expected;

  double *result = decompress_off_diagonal(&node);

  expect_arr_double_eq(result, params->matrix, node.m, node.n, node.m, params->m_full);

  free(result);
}


ParameterizedTestParameters(tree, recompress) {
  int n_params;
  struct ParametersTestCompress *params = generate_compress_params(&n_params);
  return cr_make_param_array(struct ParametersTestCompress, params, n_params, free_compress_params);
}


ParameterizedTest(struct ParametersTestCompress *params, tree, recompress) {
  struct NodeOffDiagonal node;
  int n_singular_values = params->m < params->n ? params->m : params->n;
  
  int diff = params->matrix - params->full_matrix;
  double *og_data = malloc(params->m_full * params->m_full * sizeof(double));
  memcpy(og_data, params->full_matrix, params->m_full * params->m_full * sizeof(double));
  double *og_matrix = og_data + diff;

  double *s_work = malloc(n_singular_values * sizeof(double));
  double *u_work = malloc(params->m * n_singular_values * sizeof(double));
  double *vt_work = malloc(params->n * n_singular_values * sizeof(double));

  int result_code = compress_off_diagonal(
      &node, params->m, params->n, n_singular_values, params->m_full, 
      params->matrix, s_work, u_work, vt_work, params->svd_threshold
  );

  free(s_work); free(u_work); free(vt_work);

  double *result = decompress_off_diagonal(&node);

  expect_arr_double_eq(result, og_matrix, node.m, node.n, 
                       node.m, params->m_full);

  free(og_data); free(result);
}


