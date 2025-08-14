#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../include/utils.h"
#include "../include/common_data.h"

#include "../../include/tree.h"
#include "../../include/utils.h"
#include "../../src/hodlr_algebra.c"


#define STR_LEN 10


static inline void check_n_params(const int actual, const int allocd) {
  if (actual != allocd) {
    printf("PARAMETER SET-UP FAILED - allocated %d parameters but set %d\n",
           allocd, actual);
  }
}


struct ParametersTestHxH {
  struct TreeHODLR *hodlr1;
struct TreeHODLR *hodlr2;
  struct TreeHODLR *expected;
  char hodlr1_name[STR_LEN];
  char hodlr2_name[STR_LEN];
};


void free_hh_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersTestHxH *param = 
      (struct ParametersTestHxH *) params->params + i;
    
    free_tree_hodlr(&(param->hodlr1), &cr_free);
    free_tree_hodlr(&(param->hodlr2), &cr_free);
    free_tree_hodlr(&(param->expected), &cr_free);
  }
  cr_free(params->params);
}


static void fill_laplacian_x_converse(const int m, double *matrix) {
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      if (i == j) {
        matrix[i + j * m] = -6.0;
      } else if (i == j + 1 || i == j - 1) {
        matrix[i + j * m] = 5.0;
      } else if (i == j + 2 || i == j - 2) {
        matrix[i + j * m] = -2.0;
      } else {
        matrix[i + j * m] = 0.0;
      }
    }
  }
  matrix[0] = -4.0;
  matrix[m * m - 1] = -4.0;
}


static int laplacian_matrix(struct ParametersTestHxH *params) {
  const int n_cases = 3, max_height = 3;
  int i = 0, ierr = 0;
  double svd_threshold = 1e-8;
  const int m = 21;
  double *matrix = cr_malloc(m * m * sizeof(double));

  for (int height = 1; height < max_height + 1; height++) {
    i = n_cases * (height - 1);

    for (int j = i; j < i+n_cases; j++) {
      params[j].hodlr1 = allocate_tree_monolithic(height, &ierr, 
                                                  &cr_malloc, &cr_free);
      params[j].hodlr2 = allocate_tree_monolithic(height, &ierr, 
                                                  &cr_malloc, &cr_free);
      params[j].expected = allocate_tree_monolithic(height, &ierr, 
                                                    &cr_malloc, &cr_free);
    }

    // LAPLACIAN MATRIX
    fill_laplacian_matrix(m, matrix);
    strncat(params[i].hodlr1_name, "L", STR_LEN);
    dense_to_tree_hodlr(params[i].hodlr1, m, NULL, matrix, 
                        svd_threshold, &ierr, &cr_malloc, &cr_free);

    // LAPLACIAN MATRIX with 0.5 in corners
    strncat(params[i+1].hodlr1_name, "L0.5S", STR_LEN);
    fill_laplacian_matrix(m, matrix);
    matrix[m - 1] = 0.5;
    matrix[m * (m - 1)] = 0.5;
    dense_to_tree_hodlr(params[i+1].hodlr1, m, NULL,
                        matrix, svd_threshold, &ierr, &cr_malloc, &cr_free);

    // LAPLACIAN MATRIX with 0.5 in bottom corner
    strncat(params[i+2].hodlr1_name, "L0.5A", STR_LEN);
    fill_laplacian_matrix(m, matrix);
    matrix[m - 1] = 0.5;
    dense_to_tree_hodlr(params[i+2].hodlr1, m, NULL,
                        matrix, svd_threshold, &ierr, &cr_malloc, &cr_free);

    for (int j = 0; j < n_cases; j++) {
      strncat(params[i+j].hodlr2_name, "LC", STR_LEN);
      fill_laplacian_converse_matrix(m, matrix);
      dense_to_tree_hodlr(params[i+j].hodlr2, m, NULL, matrix, svd_threshold,
                          &ierr, &cr_malloc, &cr_free);
    }

    // LAPLACIAN MATRIX
    fill_laplacian_x_converse(m, matrix);
    dense_to_tree_hodlr(params[i].expected, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);

    // LAPLACIAN MATRIX with 0.5 in corners
    i += 1;
    fill_laplacian_x_converse(m, matrix);
    matrix[m - 1] = -0.5;
    matrix[2 * m - 1] = 1.0;
    matrix[(m - 2) * m] = 1.0;
    matrix[(m - 1) * m] = -0.5;
    dense_to_tree_hodlr(params[i].expected, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);

    // LAPLACIAN MATRIX with 0.5 in bottom corner
    i += 1;
    fill_laplacian_x_converse(m, matrix);
    matrix[m - 1] = -0.5;
    matrix[2 * m - 1] = 1.0;
    dense_to_tree_hodlr(params[i].expected, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);
  }

  return n_cases * max_height;
}


static int identity_matrix(struct ParametersTestHxH *params) {
  const int n_cases = 3, max_height = 3;
  int i = 0, idx = 0, ierr = 0;
  double svd_threshold = 1e-8;
  const int m = 21;
  double *matrix = cr_malloc(m * m * sizeof(double));

  for (int height = 1; height < max_height + 1; height++) {
    idx = n_cases * (height - 1);

    for (i = idx; i < idx+n_cases; i++) {
      strncat(params[i].hodlr1_name, "I", STR_LEN);

      params[i].hodlr1 = allocate_tree_monolithic(height, &ierr,
                                                  &cr_malloc, &cr_free);
      fill_identity_matrix(m, matrix);
      dense_to_tree_hodlr(params[i].hodlr1, m, NULL, matrix, 
                          svd_threshold, &ierr, &cr_malloc, &cr_free);
      
      params[i].hodlr2 = allocate_tree_monolithic(height, &ierr,
                                                  &cr_malloc, &cr_free);
      params[i].expected = allocate_tree_monolithic(height, &ierr,
                                                    &cr_malloc, &cr_free);
    }

    strncat(params[idx].hodlr2_name, "I", STR_LEN);
    fill_identity_matrix(m, matrix);
    dense_to_tree_hodlr(params[idx].hodlr2, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);
    fill_identity_matrix(m, matrix);
    dense_to_tree_hodlr(params[idx].expected, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);

    strncat(params[idx+1].hodlr2_name, "0", STR_LEN);
    fill_full_matrix(m, 0.0, matrix);
    dense_to_tree_hodlr(params[idx+1].hodlr2, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);
    fill_full_matrix(m, 0.0, matrix);
    dense_to_tree_hodlr(params[idx+1].expected, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);

    strncat(params[idx+2].hodlr2_name, "L", STR_LEN);
    fill_laplacian_matrix(m, matrix);
    dense_to_tree_hodlr(params[idx+2].hodlr2, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);
    fill_laplacian_matrix(m, matrix);
    dense_to_tree_hodlr(params[idx+2].expected, m, NULL, matrix, svd_threshold,
                        &ierr, &cr_malloc, &cr_free);
  }

  return n_cases * max_height;
}


struct ParametersTestHxH * generate_hodlr_hodlr_params(int * len) {
  const int n_params = 9+9;
  int actual = 0;
  *len = n_params;
  struct ParametersTestHxH *params = 
    cr_malloc(n_params * sizeof(struct ParametersTestHxH));

  actual += laplacian_matrix(params);
  actual += identity_matrix(params + actual);

  check_n_params(actual, n_params);

  return params;
}


ParameterizedTestParameters(hodlr_hodlr_algebra, compute_diagonal) {
  int n_params;
  struct ParametersTestHxH *params = generate_hodlr_hodlr_params(&n_params);

  return cr_make_param_array(struct ParametersTestHxH, params, n_params, 
                             free_hh_params);
}


ParameterizedTest(struct ParametersTestHxH *params, hodlr_hodlr_algebra, 
                  compute_diagonal) {
  cr_log_info("%.10s (height=%d) x %.10s (height=%d)",
              params->hodlr1_name, params->hodlr1->height, params->hodlr2_name, 
              params->hodlr2->height);

  int ierr = 0;
  int *offsets = calloc(params->expected->height, sizeof(int));
  
  int m, highest_m = 0;
  for (int i = 0; i < params->hodlr2->len_work_queue * 2; i++) {
    m = params->hodlr2->innermost_leaves[i]->data.diagonal.m;
    if (m > highest_m) highest_m = m;
  }
  int s1 = get_highest_s(params->hodlr1);
  int s2 = get_highest_s(params->hodlr2);
  double *workspace = malloc(s1 * s2 * sizeof(double));
  double *workspace2 = malloc(s1 * highest_m * sizeof(double));

  struct TreeHODLR *result = allocate_tree_monolithic(
    params->expected->height, &ierr, &malloc, &free
  );

  struct HODLRInternalNode **queue = result->work_queue;

  compute_diagonal(
    params->hodlr1, params->hodlr2, result, queue, offsets, workspace, 
    workspace2, &ierr
  );
  if (ierr != SUCCESS) {
    cr_fail("Returned ierr (%d) different from SUCCESS (%d)", ierr, SUCCESS);
  }

  struct NodeDiagonal *r, *e;
  for (int i = 0; i < result->len_work_queue * 2; i++) {
    r = &result->innermost_leaves[i]->data.diagonal;
    e = &params->expected->innermost_leaves[i]->data.diagonal;

    expect_matrix_double_eq_safe(
      r->data, e->data, r->m, r->m, e->m, e->m, r->m, e->m, i, "", NULL, NULL
    );
  }

  free_tree_hodlr(&result, &free);
  free(offsets); free(workspace); free(workspace2);
}


ParameterizedTestParameters(hodlr_hodlr_algebra, compute_inner_off_diagonal) {
  int n_params;
  struct ParametersTestHxH *params = generate_hodlr_hodlr_params(&n_params);

  return cr_make_param_array(struct ParametersTestHxH, params, n_params, 
                             free_hh_params);
}


ParameterizedTest(struct ParametersTestHxH *params, hodlr_hodlr_algebra, 
                  compute_inner_off_diagonal) {
  cr_log_info("%.10s (height=%d) x %.10s (height=%d)",
              params->hodlr1_name, params->hodlr1->height, params->hodlr2_name, 
              params->hodlr2->height);

  int ierr = SUCCESS; const double svd_threshold = 1e-8;
  int *offsets = calloc(params->expected->height, sizeof(int));
  
  int s1 = get_highest_s(params->hodlr1);
  int s2 = get_highest_s(params->hodlr2);
  double *workspace = malloc(s1 * s2 * sizeof(double));

  struct TreeHODLR *result = allocate_tree_monolithic(
    params->expected->height, &ierr, &malloc, &free
  );
  copy_block_sizes(params->expected, result, false);

  const size_t sbuff = 50 * sizeof(char);
  char *buffer = malloc(sbuff);

  for (int parent = 0; parent < result->len_work_queue; parent++) {
    struct NodeOffDiagonal *top_right = 
      &result->innermost_leaves[2 * parent]->parent->children[1].leaf->
        data.off_diagonal;
    struct NodeOffDiagonal *bottom_left = 
      &result->innermost_leaves[2 * parent]->parent->children[2].leaf->
        data.off_diagonal;

    compute_inner_off_diagonal(
      result->height, parent, 
      params->hodlr1->innermost_leaves[2 * parent]->parent,
      params->hodlr2->innermost_leaves[2 * parent]->parent,
      top_right, bottom_left,
      svd_threshold, offsets, workspace, &ierr
    );
  
    if (ierr != SUCCESS) {
      cr_fail("Returned ierr (%d) different from SUCCESS (%d)", ierr, SUCCESS);
    }

    snprintf(buffer, sbuff, "idx=%d top right", parent);
    expect_off_diagonal(
      top_right,
      &params->expected->innermost_leaves[2 * parent]->parent->children[1].leaf->data.off_diagonal,
      buffer
    );
    snprintf(buffer, sbuff, "idx=%d bottom left", parent);
    expect_off_diagonal(
      bottom_left,
      &params->expected->innermost_leaves[2 * parent]->parent->children[2].leaf->data.off_diagonal,
      buffer
    );
  }

  free_tree_hodlr(&result, &free);
  free(offsets); free(workspace);
}


struct ParametersSComponent {
  struct HODLRInternalNode *parent;
  int height;
  int parent_position;
  int expected;
  int which_free;
};


void free_sc_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersSComponent *param = 
      (struct ParametersSComponent *) params->params + i;

    cr_free(param->parent->children[param->which_free].leaf);
    cr_free(param->parent);
  }
  cr_free(params->params);
}


struct ContextSC {
  int *restrict arr;
  const int *restrict heights;
  int idx;
};


static inline int * stack(struct ContextSC *context, int values[]) {
  const int len = context->heights[context->idx];

  for (int i = 0; i < len; i++) {
    context->arr[i] = values[i];
  }

  int *arr = context->arr;
  context->arr = context->arr + len;
  context->idx++;

  return arr;
}


ParameterizedTestParameters(hodlr_hodlr_algebra, s_component) {
  enum {n_params = 9};
  struct ParametersSComponent *params = 
    cr_malloc(n_params * sizeof(struct ParametersSComponent));

  // These go into the parameters
  const int heights[n_params] = {0, 1, 1, 2, 2, 4, 4, 4, 4};
  const int parent_position[n_params] = {0, 0, 1, 1, 2, 0, 3, 4, 15};
  const int expected[n_params] = {0, 1, 11, 2, 22, 4, 42, 9, 13};

  // Following are used to set up the node structure
  int total_height = heights[0];
  for (int i = 1; i < n_params; i++) {
    total_height += heights[i];
  }

  int *which_arr = cr_malloc(total_height * sizeof(int));
  struct ContextSC context_which = {which_arr, &heights[0], 1};
  const int *which_child[n_params] = {
    &which_arr[0],
    stack(&context_which, (int[]){1}),
    stack(&context_which, (int[]){2}),
    stack(&context_which, (int[]){2, 1}),
    stack(&context_which, (int[]){1, 2}),
    stack(&context_which, (int[]){1, 1, 1, 1}),
    stack(&context_which, (int[]){2, 2, 1, 1}),
    stack(&context_which, (int[]){1, 1, 2, 1}),
    stack(&context_which, (int[]){2, 2, 2, 2}),
  };

  int *s_arr = cr_malloc(total_height * sizeof(int));
  struct ContextSC context_s = {s_arr, &heights[0], 1};
  const int *ss[n_params] = {
    &s_arr[0],
    stack(&context_s, (int[]){1}),
    stack(&context_s, (int[]){11}),
    stack(&context_s, (int[]){1, 1}),
    stack(&context_s, (int[]){1, 21}),
    stack(&context_s, (int[]){1, 1, 1, 1}),
    stack(&context_s, (int[]){11, 11, 5, 15}),
    stack(&context_s, (int[]){2, 3, 1, 3}),
    stack(&context_s, (int[]){1, 1, 10, 1}),
  };

  struct HODLRInternalNode *temp = NULL;
  for (int i = 0; i < n_params; i++) {
    params[i].height = heights[i];
    params[i].parent_position = parent_position[i];
    params[i].expected = expected[i];
    params[i].which_free = which_child[i][0];

    params[i].parent = 
      cr_malloc(heights[i] * sizeof(struct HODLRInternalNode));
    params[i].parent->children[which_child[i][0]].leaf = 
      cr_malloc(heights[i] * sizeof(struct HODLRLeafNode));

    temp = &params[i].parent[0];
    for (int h = 1; h < heights[i]; h++) {
      temp->children[which_child[i][h-1]].leaf->data.off_diagonal.s = 
        ss[i][h-1];

      temp->parent = &params[i].parent[h];
      temp = temp->parent;
      temp->children[which_child[i][h]].leaf = 
        &params[i].parent->children[which_child[i][0]].leaf[h];
    }

    if (heights[i] > 0) {
      temp->parent = NULL;
      temp->children[which_child[i][heights[i]-1]].leaf->data.off_diagonal.s =
        ss[i][heights[i]-1];
    }
  }

  cr_free(which_arr); cr_free(s_arr);

  return cr_make_param_array(struct ParametersSComponent, params, n_params,
                             free_sc_params);
}


ParameterizedTest(struct ParametersSComponent *params, hodlr_hodlr_algebra, 
                  s_component) {
  cr_log_info("height=%d, parent_position=%d, expected=%d", 
              params->height, params->parent_position, params->expected);

  int actual = compute_workspace_size_s_component(
    params->parent, params->height, params->parent_position
  );

  cr_assert(eq(int, actual, params->expected));
}


struct ParametersHigherContribOffDiag {
  int height;
  int origin_idx;
  int divisor;
  struct TreeHODLR *hodlr_left;
  struct TreeHODLR *hodlr_right;
  struct HODLRInternalNode *parent_left;
  struct HODLRInternalNode *parent_right;
  struct NodeOffDiagonal *expected_tr;
  struct NodeOffDiagonal *expected_bl;
  int *restrict offsets;
  int *restrict expected_offsets;
  int expected_offset_utr;
  int expected_offset_vtr;
};


static void free_hcod_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersHigherContribOffDiag *param = 
      (struct ParametersHigherContribOffDiag *) params->params + i;

    free_tree_hodlr(&param->hodlr_left, &cr_free);
    free_tree_hodlr(&param->hodlr_right, &cr_free);
    cr_free(param->offsets);
    cr_free(param->expected_offsets);

    cr_free(param->expected_tr->u);
    cr_free(param->expected_tr);

    cr_free(param->expected_bl->u);
    cr_free(param->expected_bl);
  } 
  cr_free(params->params);
}


static inline int generate_hcod_params(
  struct ParametersHigherContribOffDiag *params
) {
  enum {n_params = 5};
  const int height = 5, m = 67; const double svd_threshold = 1e-8;
  int ierr;
  double *matrix = cr_malloc(m * m * sizeof(double));

  const int heights[n_params] = {0, 1, 2, 3, 4};
  const int origins[n_params] = {0, 0, 0, 2, 13};
  const int divs[n_params] = {1, 1, 1, 1, 1};

  const int ms[n_params] = {34, 17, 9, 4, 2};
  const int ss[n_params] = {0, 1, 2, 3, 4};
  const int ns[n_params] = {33, 17, 8, 4, 2};

  for (int i = 0; i < n_params; i++) {
    params[i].height = heights[i];
    params[i].origin_idx = origins[i];
    params[i].divisor = divs[i];

    params[i].hodlr_left = 
      allocate_tree_monolithic(height, &ierr, &cr_malloc, &cr_free);
    params[i].hodlr_right = 
      allocate_tree_monolithic(height, &ierr, &cr_malloc, &cr_free);

    fill_full_matrix(m, 0.0, matrix);
    dense_to_tree_hodlr(params[i].hodlr_left, m, NULL, matrix, svd_threshold, 
                        &ierr, &cr_malloc, &cr_free);
    fill_laplacian_matrix(m, matrix);
    dense_to_tree_hodlr(params[i].hodlr_right, m, NULL, matrix, svd_threshold, 
                        &ierr, &cr_malloc, &cr_free);

    params[i].offsets = cr_calloc(height, sizeof(int));
    params[i].expected_offsets = cr_calloc(height, sizeof(int));

    params[i].expected_tr = cr_malloc(sizeof(struct NodeOffDiagonal));
    params[i].expected_tr->m = ms[i];
    params[i].expected_tr->s = ss[i];
    params[i].expected_tr->n = ns[i];
    params[i].expected_tr->u = cr_calloc(ms[i] * ns[i], sizeof(double));

    params[i].expected_bl = cr_malloc(sizeof(struct NodeOffDiagonal));
    params[i].expected_bl->m = ns[i];
    params[i].expected_bl->s = ss[i];
    params[i].expected_bl->n = ms[i];
    params[i].expected_bl->u = cr_calloc(ns[i] * ns[i], sizeof(double));

    params[i].expected_offset_utr = heights[i] * (ms[i] * 1);
    params[i].expected_offset_vtr = heights[i] * (ns[i] * 1);

    for (int j = 0; j < heights[i]; j++) {
      params[i].offsets[j] = -50;
      params[i].expected_offsets[j] = ms[i] + ns[i];
    }
  }

  int i = 0;
  params[i].parent_left = params[i].hodlr_left->root->parent;
  params[i].parent_right = params[i].hodlr_right->root->parent;
  i++;

  params[i].parent_left = params[i].hodlr_left->root;
  params[i].parent_right = params[i].hodlr_right->root;
  i++;

  params[i].parent_left = params[i].hodlr_left->root->children[0].internal;
  params[i].parent_right = params[i].hodlr_right->root->children[0].internal;
  i++;

  params[i].parent_left = 
    params[i].hodlr_left->root->children[0].internal->children[3].internal;
  params[i].parent_right = 
    params[i].hodlr_right->root->children[0].internal->children[3].internal;
  params[i].offsets[2] = 17;
  params[i].expected_offsets[2] += 17;
  i++;

  params[i].parent_left = 
    params[i].hodlr_left->root->children[3].internal->children[3].internal->
      children[0].internal;
  params[i].parent_right = 
    params[i].hodlr_right->root->children[3].internal->children[3].internal->
      children[0].internal;
  params[i].offsets[1] = 4;
  params[i].expected_offsets[1] += 4;
  params[i].offsets[2] = 4;
  params[i].expected_offsets[2] += 4;
  params[i].offsets[3] = 17+4;
  params[i].expected_offsets[3] += 17+4;
  i++;

  check_n_params(i, n_params);

  return n_params;
}


ParameterizedTestParameters(hodlr_hodlr_algebra, 
                            compute_higher_level_contributions_off_diagonal) {
  enum {n_params = 5};
  struct ParametersHigherContribOffDiag *params = 
    cr_malloc(n_params * sizeof(struct ParametersHigherContribOffDiag));

  int actual = generate_hcod_params(params);

  check_n_params(actual, n_params);

  return cr_make_param_array(struct ParametersHigherContribOffDiag, params, 
                             n_params, free_hcod_params);
}


static inline struct NodeOffDiagonal * init_actual(
  const struct NodeOffDiagonal *expected
) {
  struct NodeOffDiagonal *actual = malloc(sizeof(struct NodeOffDiagonal));

  actual->m = expected->m;
  actual->s = expected->s;
  actual->n = expected->n;

  actual->u = malloc(actual->m * actual->s * sizeof(double));
  for (int j = 0; j < actual->s; j++) {
    for (int i = 0; i < actual->m; i++) {
      actual->u[i + j * actual->m] = 500.0;
    }
  }

  actual->v = malloc(actual->n * actual->s * sizeof(double));
  for (int j = 0; j < actual->s; j++) {
    for (int i = 0; i < actual->n; i++) {
      actual->v[i + j * actual->n] = 500.0;
    }
  }

  return actual;
}


ParameterizedTest(struct ParametersHigherContribOffDiag *params, 
                  hodlr_hodlr_algebra, 
                  compute_higher_level_contributions_off_diagonal) {
  cr_log_info("height=%d, origin_idx=%d, m=%d, n=%d, s=%d",
              params->height, params->origin_idx, params->expected_tr->m,
              params->expected_tr->n, params->expected_tr->s);

  struct NodeOffDiagonal *actual_tr = init_actual(params->expected_tr);
  struct NodeOffDiagonal *actual_bl = init_actual(params->expected_bl);

  int actual_offset_utr = -1, actual_offset_vtr = -1;
  double *workspace = malloc(actual_tr->s * actual_bl->s * sizeof(double));

  compute_higher_level_contributions_off_diagonal(
    params->height, params->origin_idx, params->divisor, params->parent_left,
    params->parent_right, actual_tr, actual_bl, params->offsets, workspace,
    &actual_offset_utr, &actual_offset_vtr
  );
  free(workspace);

  cr_expect(eq(int, actual_offset_utr, params->expected_offset_utr));
  cr_expect(eq(int, actual_offset_vtr, params->expected_offset_vtr));

  for (int i = 0; i < params->hodlr_left->height; i++) {
    cr_expect(eq(int, params->offsets[i], params->expected_offsets[i]),
              "offset idx=%d", i);
  }

  double *workspace2 = malloc(actual_tr->m * actual_tr->n * sizeof(double)); 
  expect_off_diagonal_recompress(
    actual_tr, params->expected_tr, actual_tr->m, workspace2
  );
  expect_off_diagonal_recompress(
    actual_bl, params->expected_bl, actual_bl->m, workspace2
  );

  free(actual_tr->u); free(actual_tr->v); free(actual_tr); 
  free(actual_bl->u); free(actual_bl->v); free(actual_bl); 
  free(workspace2); 
}


struct ParametersInnerOffDiagLowest {
  struct NodeDiagonal *diagonal_left;
  struct NodeOffDiagonal *off_diagonal_left;
  struct NodeDiagonal *diagonal_right;
  struct NodeOffDiagonal *off_diagonal_right;
  struct NodeOffDiagonal *expected;
  int offset_u;
  int offset_v;
};


void free_iodl_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersInnerOffDiagLowest *param = 
      (struct ParametersInnerOffDiagLowest *) params->params + i;

    cr_free(param->diagonal_left->data);
    cr_free(param->diagonal_left);
    cr_free(param->diagonal_right->data);
    cr_free(param->diagonal_right);

    cr_free(param->off_diagonal_left->u);
    cr_free(param->off_diagonal_left->v);
    cr_free(param->off_diagonal_left);

    cr_free(param->off_diagonal_right->u);
    cr_free(param->off_diagonal_right->v);
    cr_free(param->off_diagonal_right);

    cr_free(param->expected);
  } 
  cr_free(params->params);
}


ParameterizedTestParameters(hodlr_hodlr_algebra, 
                            compute_inner_off_diagonal_lowest_level) {
  enum {n_params = 1};
  struct ParametersInnerOffDiagLowest *params = 
    cr_malloc(n_params * sizeof(struct ParametersInnerOffDiagLowest));

  int i = 0;
  params[i].offset_u = 0; params[i].offset_v = 0;

  params[i].diagonal_left = cr_malloc(sizeof(struct NodeDiagonal));
  params[i].diagonal_left->m = 10;
  params[i].diagonal_left->data = 
    cr_calloc(params[i].diagonal_left->m, sizeof(double));
  params[i].diagonal_right = cr_malloc(sizeof(struct NodeDiagonal));
  params[i].diagonal_right->m = 9;
  params[i].diagonal_right->data = 
    cr_calloc(params[i].diagonal_left->m, sizeof(double));

  params[i].off_diagonal_left = cr_malloc(sizeof(struct NodeOffDiagonal));
  params[i].off_diagonal_left->m = params[i].diagonal_left->m;
  params[i].off_diagonal_left->s = 1;
  params[i].off_diagonal_left->n = params[i].diagonal_right->m;
  params[i].off_diagonal_left->u = 
    cr_calloc(params[i].off_diagonal_left->m * params[i].off_diagonal_left->s,
              sizeof(double));
  params[i].off_diagonal_left->v = 
    cr_calloc(params[i].off_diagonal_left->n * params[i].off_diagonal_left->s,
              sizeof(double));

  params[i].off_diagonal_right = cr_malloc(sizeof(struct NodeOffDiagonal));
  params[i].off_diagonal_right->m = params[i].diagonal_left->m;
  params[i].off_diagonal_right->s = 1;
  params[i].off_diagonal_right->n = params[i].diagonal_right->m;
  params[i].off_diagonal_right->u = 
    cr_calloc(params[i].off_diagonal_right->m * params[i].off_diagonal_right->s,
              sizeof(double));
  params[i].off_diagonal_right->v = 
    cr_calloc(params[i].off_diagonal_right->n * params[i].off_diagonal_right->s,
              sizeof(double));

  params[i].expected = cr_malloc(sizeof(struct NodeOffDiagonal));
  params[i].expected->m = params[i].diagonal_left->m;
  params[i].expected->s = 
    params[i].off_diagonal_left->s + params[i].off_diagonal_right->s;
  params[i].expected->n = params[i].diagonal_right->m;
  params[i].expected->u = 
    cr_calloc(params[i].expected->m * params[i].expected->s,
              sizeof(double));
  params[i].expected->v = 
    cr_calloc(params[i].expected->n * params[i].expected->s,
              sizeof(double));

  params[i].off_diagonal_left->u[params[i].off_diagonal_left->m - 1] = 5.0;
  params[i].off_diagonal_left->v[0] = 1.0;
  params[i].off_diagonal_right->u[0] = -1.0;
  params[i].off_diagonal_right->v[params[i].off_diagonal_right->n - 1] = 4.2;

  memcpy(params[i].expected->u + params[i].expected->m * params[i].off_diagonal_right->s,
         params[i].off_diagonal_left->u, 
         params[i].expected->m * params[i].off_diagonal_left->s * sizeof(double));

  memcpy(params[i].expected->v, params[i].off_diagonal_right->v, 
         params[i].expected->n * params[i].off_diagonal_right->s * sizeof(double));

  return cr_make_param_array(struct ParametersInnerOffDiagLowest, params, 
                             n_params, free_iodl_params);
}


ParameterizedTest(struct ParametersInnerOffDiagLowest *params, 
                  hodlr_hodlr_algebra, 
                  compute_inner_off_diagonal_lowest_level) {
  struct NodeOffDiagonal result;
  result.m = params->diagonal_left->m;
  result.s = params->off_diagonal_left->s + params->off_diagonal_right->s;
  result.n = params->diagonal_right->m;
  result.u = malloc(result.m * result.s * sizeof(double));
  result.v = malloc(result.n * result.s * sizeof(double));

  compute_inner_off_diagonal_lowest_level(
    params->diagonal_left, params->off_diagonal_left, params->diagonal_right, 
    params->off_diagonal_right, &result, params->offset_u, params->offset_v
  );

  expect_off_diagonal(&result, params->expected, "");

  free(result.u); free(result.v);
}


struct ParametersRecompress {
  struct NodeOffDiagonal *restrict node;
  struct NodeOffDiagonal *restrict expected;
  double svd_threshold;
};


void free_recompress_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersRecompress *param = 
      (struct ParametersRecompress *) params->params + i;
    
    cr_free(param->node); 
    cr_free(param->expected);
  }
  cr_free(params->params);
}


ParameterizedTestParameters(hodlr_hodlr_algebra, recompress) {
  enum {n_params = 4};
  struct ParametersRecompress *params = 
    cr_malloc(n_params * sizeof(struct ParametersRecompress));

  const int ms[n_params] = {10, 10, 9, 6};
  const int ss[n_params] = {2, 3, 4, 2};
  const int ns[n_params] = {10, 9, 10, 6};
  const int ss_exp[n_params] = {1, 1, 1, 1};

  for (int i = 0; i < n_params; i++) {
    params[i].node = cr_malloc(sizeof(struct NodeOffDiagonal));
    params[i].expected = cr_malloc(sizeof(struct NodeOffDiagonal));

    params[i].node->m = ms[i]; params[i].expected->m = ms[i];
    params[i].node->n = ns[i]; params[i].expected->n = ns[i];
    params[i].node->s = ss[i];
    params[i].expected->s = ss_exp[i];

    params[i].node->u = cr_calloc(ms[i] * ss[i], sizeof(double));
    params[i].node->v = cr_calloc(ns[i] * ss[i], sizeof(double));

    params[i].expected->u = cr_calloc(ms[i] * ss_exp[i], sizeof(double));
    params[i].expected->v = cr_calloc(ns[i] * ss_exp[i], sizeof(double));
  }

  int i = 0;
  params[i].node->u[0] = 1.0;
  params[i].node->v[0] = 1.0;
  params[i].expected->u[0] = 1.0;
  params[i].expected->v[0] = 1.0;
  i++;

  params[i].node->u[params[i].node->m - 1] = 1.0;
  params[i].node->v[0] = 1.0;
  params[i].expected->u[params[i].node->m - 1] = -1.0;
  params[i].expected->v[0] = -1.0;
  i++;

  params[i].node->u[0] = 1.0;
  params[i].node->v[params[i].node->n - 1] = 1.0;
  params[i].expected->u[0] = 1.0;
  params[i].expected->v[params[i].node->n - 1] = 1.0;
  i++;

  params[i].node->u[0] = 1.0;
  params[i].node->u[params[i].node->m] = 1.0;
  params[i].node->v[params[i].node->n - 1] = 1.0;
  params[i].node->v[2 * params[i].node->n - 1] = 1.0;
  params[i].expected->u[0] = 2.0;
  params[i].expected->v[params[i].node->n - 1] = 1.0;
  i++;


  check_n_params(i, n_params);

  return cr_make_param_array(struct ParametersRecompress, params, n_params, 
                             free_recompress_params);
}


static inline struct NodeOffDiagonal * copy_node(
  const struct NodeOffDiagonal *const src
) {
  struct NodeOffDiagonal *new = malloc(sizeof(struct NodeOffDiagonal));

  new->m = src->m; new->n = src->n; new->s = src->s;

  new->u = malloc(new->m * new->s * sizeof(double));
  memcpy(new->u, src->u, new->m * new->s * sizeof(double));

  new->v = malloc(new->n * new->s * sizeof(double));
  memcpy(new->v, src->v, new->n * new->s * sizeof(double));

  return new;
}


ParameterizedTest(struct ParametersRecompress *params, hodlr_hodlr_algebra, 
                  recompress) {
  int ierr = SUCCESS; 
  const double svd_threshold = 1e-8;

  int m_larger, m_smaller;
  if (params->node->m > params->node->n) {
    m_larger = params->node->m; m_smaller = params->node->n;
  } else {
    m_larger = params->node->n; m_smaller = params->node->m;
  }

  struct NodeOffDiagonal *node = copy_node(params->node);

  int result = recompress(node, m_larger, m_smaller, svd_threshold, &ierr);

  cr_expect(eq(int, ierr, SUCCESS));
  cr_expect(eq(int, result, 0));
  if (ierr != SUCCESS || result != 0) {
    free(node->u); free(node->v); free(node);
    cr_fatal();
  }

  expect_off_diagonal(node, params->expected, "");

  free(node->u); free(node->v); free(node);
}


struct ParametersOffDiagContrib {
  struct NodeOffDiagonal *restrict leaf1;
  struct NodeOffDiagonal *restrict leaf2;
  int offset;
  double *restrict out;
  double *restrict expected;
  int m;
  char out_name;
};


void free_odc_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersOffDiagContrib *param = 
      (struct ParametersOffDiagContrib *) params->params + i;
    
    cr_free(param->leaf1); cr_free(param->leaf2);
    cr_free(param->out); cr_free(param->expected);
  }
  cr_free(params->params);
}


static inline double * zeros(const int m) {
  return cr_calloc(m * m, sizeof(double));
}


static inline double * ones(const int m) {
  return construct_full_matrix(m, 1.0);
}


int create_params_add_off_diag_contrib(
  struct ParametersOffDiagContrib *params,
  double *(*func)(int),
  char out_name
) {
  enum {n_params = 7};

  const int ms[n_params] = {5, 5, 5, 5, 5, 10, 10};
  const int mms[n_params] = {5, 10, 10, 15, 15, 10, 10};
  const int ns[n_params] = {5, 10, 10, 15, 15, 9, 11};
  const int s1s[n_params] = {1, 1, 1, 1, 1, 3, 3};
  const int s2s[n_params] = {1, 1, 1, 1, 1, 2, 2};
  const int offsets[n_params] = {0, 5, 0, 5, 0, 0, 0};

  for (int i = 0; i < n_params; i++) {
    params[i].leaf1 = cr_malloc(sizeof(struct NodeOffDiagonal));
    params[i].leaf1->m = mms[i];
    params[i].leaf1->n = ns[i];
    params[i].leaf1->s = s1s[i];
    params[i].leaf1->u = cr_calloc(mms[i] * s1s[i], sizeof(double));
    params[i].leaf1->v = cr_calloc(ns[i] * s1s[i], sizeof(double));

    params[i].leaf2 = cr_malloc(sizeof(struct NodeOffDiagonal));
    params[i].leaf2->m = ns[i];
    params[i].leaf2->n = mms[i];
    params[i].leaf2->s = s2s[i];
    params[i].leaf2->u = cr_calloc(ns[i] * s2s[i], sizeof(double));
    params[i].leaf2->v = cr_calloc(mms[i] * s2s[i], sizeof(double));

    params[i].m = ms[i];
    params[i].out = func(ms[i]);
    params[i].out_name = out_name;
    params[i].expected = func(ms[i]);
    params[i].offset = offsets[i];
  }

  for (int i = 0; i < 4; i++) {
    params[i].leaf1->u[params[i].leaf1->m - 1] = 1.0;
    params[i].leaf1->v[0] = 1.0;
    params[i].leaf2->u[0] = 1.0;
    params[i].leaf2->v[params[i].leaf2->m - 1] = 1.0;
  }

  for (int i = 0; i < 2; i++) {
    params[i].expected[params[i].m * params[i].m - 1] += 1.0;
  }

  int idx = 4;
  params[idx].leaf1->u[0] = 1.0;
  params[idx].leaf1->v[params[idx].leaf1->n - 1] = 1.0;
  params[idx].leaf2->u[params[idx].leaf2->m - 1] = 1.0;
  params[idx].leaf2->v[0] = 1.0;
  params[idx].expected[0] += 1.0;

  // More complex test:
  idx++;
  for (int i = 0; i < 2; i++) {
    params[idx].leaf1->u[0] = 1.0;
    params[idx].leaf1->u[params[idx].leaf1->m] = 1.0;
    params[idx].leaf1->u[2 * params[idx].leaf1->m + 1] = 9.0;
    params[idx].leaf1->v[params[idx].leaf1->n - 2] = 9.0;
    params[idx].leaf1->v[params[idx].leaf1->n - 1] = -0.5;
    params[idx].leaf1->v[params[idx].leaf1->n] = 0.01;
    params[idx].leaf1->v[params[idx].leaf1->n * 3 - 1] = 1.0;

    params[idx].leaf2->u[params[idx].leaf2->m - 2] = -9.0;
    params[idx].leaf2->u[params[idx].leaf2->m - 1] = 3.3;
    params[idx].leaf2->u[params[idx].leaf2->m * 2 - 1] = 6.6;
    params[idx].leaf2->v[0] = -0.25;
    params[idx].leaf2->v[1] = 1.0;
    params[idx].leaf2->v[params[idx].leaf2->n + 1] = 11.0;
    params[idx].expected[0] += 20.6625;
    params[idx].expected[1] += -7.425;
    params[idx].expected[params[idx].m] += -118.95;
    params[idx].expected[params[idx].m + 1] += 683.1;

    idx++;
  }

  return n_params;
}


ParameterizedTestParameters(hodlr_hodlr_algebra, add_off_diag_contrib) {
  const int np_per_func = 7;
  const int n_params = 3 * np_per_func;
  struct ParametersOffDiagContrib *params = 
    cr_malloc(n_params * sizeof(struct ParametersOffDiagContrib));
  int actual = 0;

  actual += create_params_add_off_diag_contrib(params, zeros, '0');
  actual += create_params_add_off_diag_contrib(params + actual, 
                                               construct_laplacian_matrix, 'L');
  actual += create_params_add_off_diag_contrib(params + actual,
                                               ones, '1');

  if (actual != n_params) {
    printf("PARAMETER SET-UP FAILED - allocated %d parameters but set %d\n",
           n_params, actual);
  }

  return cr_make_param_array(struct ParametersOffDiagContrib, params, n_params, 
                             free_odc_params);
}


ParameterizedTest(struct ParametersOffDiagContrib *params, hodlr_hodlr_algebra, 
                  add_off_diag_contrib) {
  cr_log_info("leaf1 (m=%d, s=%d, n=%d) x leaf2 (m=%d, s=%d, n=%d) + '%c' = "
              "result (%dx%d)", 
              params->leaf1->m, params->leaf1->s, params->leaf1->n,
              params->leaf2->m, params->leaf2->s, params->leaf2->n,
              params->out_name, params->m, params->m);

  double *workspace = malloc(params->leaf1->s * params->leaf2->s * sizeof(double));
  double *workspace2 = malloc(params->m * params->leaf1->s * sizeof(double));

  add_off_diagonal_contribution(
    params->leaf1, params->leaf2, params->offset, workspace, workspace2,
    params->out, params->m
  );

  expect_matrix_double_eq(
    params->out, params->expected, params->m, params->m, params->m, params->m,
    'M', NULL, NULL
  );

  free(workspace); free(workspace2);
}

