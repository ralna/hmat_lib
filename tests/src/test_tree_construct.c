#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../include/utils.h"
#include "../include/common_data.h"
#include "../include/tree_stubs.h"

#include "../../src/tree.c"
#include "../../include/error.h"
#include "../../include/blas_wrapper.h"


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
  int ierr;
  int n_singular_values = params->m < params->n ? params->m : params->n;
  
  double *s_work = malloc(n_singular_values * sizeof(double));
  double *u_work = malloc(params->m * n_singular_values * sizeof(double));
  double *vt_work = malloc(params->n * n_singular_values * sizeof(double));

  int result_code = compress_off_diagonal(
      &result, params->m, params->n, n_singular_values, params->m_full, 
      params->matrix, s_work, u_work, vt_work, params->svd_threshold, &ierr
  );

  free(s_work); free(u_work); free(vt_work);

  cr_expect(eq(ierr, SUCCESS));
  cr_expect(eq(result_code, 0));
  if (result_code != 0 || ierr != SUCCESS) {
    cr_fatal();
  } 
  cr_expect(eq(result.m, params->m));
  cr_expect(eq(result.s, params->expected_n_singular));
  cr_expect(eq(result.n, params->n));

  expect_arr_double_eq(result.u, params->u_expected, params->m, params->expected_n_singular,
                       result.m, params->m, 'U');
  expect_arr_double_eq(result.v, params->v_expected, params->n, params->expected_n_singular,
                       result.m, params->m, 'V');
}


ParameterizedTestParameters(tree, recompress) {
  int n_params;
  struct ParametersTestCompress *params = generate_compress_params(&n_params);
  return cr_make_param_array(struct ParametersTestCompress, params, n_params, free_compress_params);
}


ParameterizedTest(struct ParametersTestCompress *params, tree, recompress) {
  struct NodeOffDiagonal node;
  int ierr;
  double alpha = 1, beta = 0;

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
      params->matrix, s_work, u_work, vt_work, params->svd_threshold,
      &ierr
  );
  
  free(s_work); free(u_work); free(vt_work);
  
  cr_expect(eq(ierr, SUCCESS));
  cr_expect(eq(result_code, 0));
  if (result_code != 0 || ierr != SUCCESS) {
    free(og_data);
    cr_fatal();
  } 

  double *result = malloc(params->m * params->n * sizeof(double));
  dgemm_("N", "T", &node.m, &node.n, 
         &node.s, &alpha, node.u, &node.m, 
         node.v, &node.n, &beta, result, &params->m);

  expect_arr_double_eq(result, og_matrix, node.m, node.n, 
                       node.m, params->m_full, 'A');

  free(og_data); free(result);
}


Test(tree, allocate_fail) {
  int ierr;

  struct TreeHODLR *hodlr = allocate_tree(0, &ierr);

  cr_expect(eq(ierr, INPUT_ERROR));
  cr_expect(eq(hodlr, NULL));
}


struct ParametersTestDense {
  double *matrix;
  int m;
  int height;
  double svd_threshold;
  struct TreeHODLR *expected;
};


int expect_tree_consistent(struct TreeHODLR *hodlr, int height) {
  const int max_depth_n = (int)pow(2, height - 1);
  int len_queue = 1;

  cr_expect(eq(hodlr->root->parent, NULL));

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **temp_pointer;
  queue[0] = hodlr->root;

  for (int i = 1; i < height; i++) {
    for (int j = 0; j < len_queue; j++) {
      cr_expect(eq(queue[j]->children[1].leaf->parent, queue[j]));
      cr_expect(eq(queue[j]->children[2].leaf->parent, queue[j]));

      next_level[2 * j] = queue[j]->children[0].internal;
      next_level[2 * j + 1] = queue[j]->children[3].internal;

    }
    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;

    len_queue = len_queue * 2;
  }

  for (int i = 0; i < len_queue; i++) {
    for (int j = 0; j < 4; j ++) {
      cr_expect(eq(queue[i]->children[j].leaf->parent, queue[i]));
    }

    cr_expect(eq(hodlr->innermost_leaves[2 * i], queue[i]->children[0].leaf));
    cr_expect(eq(hodlr->innermost_leaves[2 * i + 1], queue[i]->children[3].leaf));
  }

  free(queue); free(next_level);
  return 0;
}


int expect_tree_hodlr(struct TreeHODLR *actual, struct TreeHODLR *expected) {
  cr_expect(eq(actual->height, expected->height));
  if (actual->height != expected->height) {
    return 1;
  }

  const int max_depth_n = (int)pow(2, expected->height - 1);
  int len_queue = 1, err = 0;

  union HODLRData *exp, *act;

  struct HODLRInternalNode **queue_a = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **next_level_a = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **queue_e = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **next_level_e = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **temp_pointer;
  queue_a[0] = actual->root;
  queue_e[0] = expected->root;

  for (int i = 1; i < expected->height; i++) {
    for (int j = 0; j < len_queue; j++) {
      for (int k = 1; k < 3; k++) {
        act = &(queue_a[j]->children[k].leaf->data);
        exp = &(queue_e[j]->children[k].leaf->data);

        err += expect_arr_double_eq_safe(act->off_diagonal.u, exp->off_diagonal.u,
                                         act->off_diagonal.m, act->off_diagonal.s,
                                         exp->off_diagonal.m, exp->off_diagonal.s,
                                         act->off_diagonal.m, exp->off_diagonal.m,
                                         'U');
        
        err += expect_arr_double_eq_safe(act->off_diagonal.v, exp->off_diagonal.v,
                                         act->off_diagonal.n, exp->off_diagonal.s,
                                         exp->off_diagonal.n, exp->off_diagonal.s,
                                         act->off_diagonal.n, exp->off_diagonal.n,
                                         'V');
      }

      cr_expect(eq(queue_a[j]->children[0].internal->m, queue_e[j]->children[3].internal->m));

      next_level_a[2 * j] = queue_a[j]->children[0].internal;
      next_level_a[2 * j + 1] = queue_a[j]->children[3].internal;

      next_level_e[2 * j] = queue_e[j]->children[0].internal;
      next_level_e[2 * j + 1] = queue_e[j]->children[3].internal;

    }
    temp_pointer = queue_a;
    queue_a = next_level_a;
    next_level_a = temp_pointer;

    temp_pointer = queue_e;
    queue_e = next_level_e;
    next_level_e = temp_pointer;

    len_queue = len_queue * 2;
  }

  for (int i = 0; i < len_queue; i++) {
    for (int j = 1; j < 3; j ++) {
      cr_log_info("Lowest level: i=%d j=%d", i, j);
      act = &(queue_a[i]->children[j].leaf->data);
      exp = &(queue_e[i]->children[j].leaf->data);
      err += expect_arr_double_eq_safe(act->off_diagonal.u, exp->off_diagonal.u,
                                       act->off_diagonal.m, act->off_diagonal.s,
                                       exp->off_diagonal.m, exp->off_diagonal.s,
                                       act->off_diagonal.m, exp->off_diagonal.m,
                                       'U');
      err += expect_arr_double_eq_safe(act->off_diagonal.v, exp->off_diagonal.v,
                                       act->off_diagonal.n, exp->off_diagonal.s,
                                       exp->off_diagonal.n, exp->off_diagonal.s,
                                       act->off_diagonal.n, exp->off_diagonal.n,
                                       'V');
    }
    act = &(queue_a[i]->children[0].leaf->data);
    exp = &(queue_e[i]->children[0].leaf->data);

    cr_expect(eq(act->diagonal.m, exp->diagonal.m));

    if (act->diagonal.m == exp->diagonal.m) {
      expect_arr_double_eq(act->diagonal.data, exp->diagonal.data, 
                           exp->diagonal.m, exp->diagonal.m, 
                           act->diagonal.m, exp->diagonal.m, 'M');
    }

    act = &(queue_a[i]->children[3].leaf->data);
    exp = &(queue_e[i]->children[3].leaf->data);

    cr_expect(eq(act->diagonal.m, exp->diagonal.m));

    if (act->diagonal.m == exp->diagonal.m) {
      expect_arr_double_eq(act->diagonal.data, exp->diagonal.data, 
                           exp->diagonal.m, exp->diagonal.m, 
                           act->diagonal.m, exp->diagonal.m, 'M');
    } 
  }

  free(queue_a); free(next_level_a); free(queue_e); free(next_level_e);
  return err;
}


void free_dense_params(struct criterion_test_params *params) {
  for (int i = 0; i < params->length; i++) {
    struct ParametersTestDense *param = params->params + i;
    cr_free(param->matrix);
    free_tree_hodlr_cr(&param->expected);
  }
  cr_free(params->params);
}


struct ParametersTestDense * generate_dense_params(int * len) {
  int n_params = 1;
  *len = n_params;
  struct ParametersTestDense *params = cr_malloc(n_params * sizeof(struct ParametersTestDense));

  // Set up test matrix
  params[0].height = 1;
  params[0].m = 21;
  params[0].matrix = construct_laplacian_matrix(params[0].m);
  params[0].matrix[params[0].m - 1] = 0.5;
  params[0].matrix[params[0].m * (params[0].m - 1)] = 0.5;

  // Set up HODLR
  params[0].svd_threshold = 0.1;
  params[0].expected = cr_malloc(sizeof(struct TreeHODLR));
  params[0].expected->height = params[0].height;
  params[0].expected->innermost_leaves = cr_malloc(2 * sizeof(struct NodeDiagonal *));

  // Set up the root node
  params[0].expected->root = cr_malloc(sizeof(struct HODLRInternalNode));
  params[0].expected->root->parent = NULL;

  // Allocate children
  for (int i = 0; i < 4; i++) {
    params[0].expected->root->children[i].leaf = cr_malloc(sizeof(struct HODLRLeafNode));
    params[0].expected->root->children[i].leaf->type = (i == 0 || i == 3) ? DIAGONAL : OFFDIAGONAL;
    params[0].expected->root->children[i].leaf->parent = params[0].expected->root;
  }

  int m_larger = 11, m_smaller = 10;
  params[0].expected->root->children[0].leaf->data.diagonal.data = construct_laplacian_matrix(m_larger);
  params[0].expected->root->children[3].leaf->data.diagonal.data = construct_laplacian_matrix(m_smaller);
  
  params[0].expected->root->children[0].leaf->data.diagonal.m = m_larger;
  params[0].expected->root->children[3].leaf->data.diagonal.m = m_smaller;

  for (int i = 1; i < 3; i++) {
    params[0].expected->root->children[i].leaf->data.off_diagonal.m = m_larger;
    params[0].expected->root->children[i].leaf->data.off_diagonal.s = 2;
    params[0].expected->root->children[i].leaf->data.off_diagonal.n = m_smaller;
    
    params[0].expected->root->children[i].leaf->data.off_diagonal.u = cr_calloc(m_larger * 2, sizeof(double));
    params[0].expected->root->children[i].leaf->data.off_diagonal.v = cr_calloc(m_smaller * 2, sizeof(double));
     
    m_larger -= 1; m_smaller += 1;
  }

  params[0].expected->innermost_leaves[0] = params[0].expected->root->children[0].leaf;
  params[0].expected->innermost_leaves[1] = params[0].expected->root->children[3].leaf;

  // Top right block
  params[0].expected->root->children[1].leaf->data.off_diagonal.u[10] = 1;
  params[0].expected->root->children[1].leaf->data.off_diagonal.u[11] = -0.5;

  params[0].expected->root->children[1].leaf->data.off_diagonal.v[0] = -1;
  params[0].expected->root->children[1].leaf->data.off_diagonal.v[19] = -1;

  // Bottom left block
  params[0].expected->root->children[2].leaf->data.off_diagonal.u[0] = 1;
  params[0].expected->root->children[2].leaf->data.off_diagonal.u[19] = 0.5;

  params[0].expected->root->children[2].leaf->data.off_diagonal.v[10] = -1;
  params[0].expected->root->children[2].leaf->data.off_diagonal.v[11] = 1;

  return params;
}


ParameterizedTestParameters(tree, dense_to_tree) {
  int n_params;
  struct ParametersTestDense *params = generate_dense_params(&n_params);
  return cr_make_param_array(struct ParametersTestDense, params, n_params, free_dense_params);
}


ParameterizedTest(struct ParametersTestDense *params, tree, dense_to_tree) {
  int ierr;

  struct TreeHODLR *result = allocate_tree(params->height, &ierr);
  cr_expect(eq(ierr, SUCCESS));
  cr_expect(ne(result, NULL));
  if (ierr != SUCCESS) {
    cr_fatal("Tree HODLR allocation failed");
  }

  expect_tree_consistent(result, params->height);
  
  int svd = dense_to_tree_hodlr(result, params->m, params->matrix, params->svd_threshold, &ierr);
  
  cr_expect(eq(ierr, SUCCESS));
  cr_expect(zero(svd));
  if (ierr != SUCCESS) {
    free_tree_hodlr(&result);
    cr_fatal();
  }

  expect_tree_hodlr(result, params->expected);
  free_tree_hodlr(&result);
}
