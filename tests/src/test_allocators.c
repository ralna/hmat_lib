#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../include/utils.h"

#include "../../src/allocators.c"
#include "../../include/error.h"
#include "../../include/blas_wrapper.h"


Test(allocators, initialise_leaf_offdiagonal) {
  struct HODLRLeafNode leaf;
  struct HODLRInternalNode parent;

  initialise_leaf_offdiagonal(&leaf, &parent);
  expect_leaf_offdiagonal(&leaf, &parent);
}


Test(allocators, initialise_leaf_diagonal) {
  struct HODLRLeafNode leaf;
  struct HODLRInternalNode parent;

  initialise_leaf_diagonal(&leaf, &parent);
  expect_leaf_diagonal(&leaf, &parent);
}


Test(allocators, initialise_internal) {
  struct HODLRInternalNode node;
  struct HODLRInternalNode parent;

  initialise_internal(&node, &parent);
  expect_internal(&node, &parent);
}


struct ParametersArraySizes {
  int height;

  struct {
    size_t len_work_queue;
    size_t len_internal_nodes;
    size_t len_innermost_leaves;
    size_t len_leaf_nodes;
  } expected;
};


void set_array_sizes_params(struct ParametersArraySizes *param, 
                            size_t swq, size_t sin, size_t sil, size_t sln) {
  param->expected.len_work_queue = swq;
  param->expected.len_internal_nodes = sin;
  param->expected.len_innermost_leaves = sil;
  param->expected.len_leaf_nodes = sln;
}


void free_params(struct criterion_test_params *params) {
  cr_free(params->params);
}


struct ParametersArraySizes * generate_array_sizes_params(int *n_params) {
  *n_params = 6;
  struct ParametersArraySizes *params = 
    cr_malloc(*n_params * sizeof(struct ParametersArraySizes));

  for (int i = 0; i < *n_params; i++) {
    params[i].height = i + 1;
  }

  set_array_sizes_params(&params[0], 1, 1, 2, 4);
  set_array_sizes_params(&params[1], 2, 3, 4, 10);
  set_array_sizes_params(&params[2], 4, 7, 8, 22);
  set_array_sizes_params(&params[3], 8, 15, 16, 46);
  set_array_sizes_params(&params[4], 16, 31, 32, 94);
  set_array_sizes_params(&params[5], 32, 63, 64, 190);

  return params;
}


ParameterizedTestParameters(allocators, test_array_sizes) {
  int n_params = 6;
  struct ParametersArraySizes *params = generate_array_sizes_params(&n_params);
  return cr_make_param_array(struct ParametersArraySizes, 
                             params, n_params, free_params);
}


ParameterizedTest(struct ParametersArraySizes *params, allocators, 
                  test_array_sizes) {
  size_t size_work_queue = 0, size_internal_nodes = 0;
  size_t size_innermost_leaves = 0, size_leaf_nodes = 0, expected_size = 0;

  cr_log_info("height=%d", params->height);

  compute_construct_tree_array_sizes(
    params->height, &size_internal_nodes, &size_leaf_nodes,
    &size_work_queue, &size_innermost_leaves
  );

  expected_size = params->expected.len_work_queue * sizeof(struct HODLRInternalNode *);
  cr_expect(eq(sz, size_work_queue, expected_size),
            "size_work_queue: actual=%ld != expected=%ld", 
            size_work_queue, expected_size);

  expected_size = params->expected.len_internal_nodes * sizeof(struct HODLRInternalNode);
  cr_expect(eq(sz, size_internal_nodes, expected_size),
            "size_internal_nodes: actual=%ld != expected=%ld", 
            size_internal_nodes, expected_size);

  expected_size = params->expected.len_innermost_leaves * sizeof(struct HODLRLeafNode *);
  cr_expect(eq(sz, size_innermost_leaves, expected_size),
            "size_innermost_leaves: actual=%ld != expected=%ld", 
            size_innermost_leaves, expected_size);

  expected_size = params->expected.len_leaf_nodes * sizeof(struct HODLRLeafNode);
  cr_expect(eq(sz, size_leaf_nodes, expected_size),
            "size_leaf_nodes: actual=%ld != expected=%ld", 
            size_leaf_nodes, expected_size);
}


ParameterizedTestParameters(allocators, test_construct_tree) {
  int n_params = 6;
  struct ParametersArraySizes *params = generate_array_sizes_params(&n_params);
  return cr_make_param_array(struct ParametersArraySizes, 
                             params, n_params, free_params);
}


ParameterizedTest(struct ParametersArraySizes *params, allocators, 
                  test_construct_tree) {
  cr_log_info("height=%d", params->height);
  int ierr = SUCCESS;

  struct TreeHODLR *hodlr = malloc(sizeof(struct TreeHODLR));
  struct HODLRInternalNode *internal_nodes = 
    malloc(params->expected.len_internal_nodes * sizeof(struct HODLRInternalNode));
  struct HODLRLeafNode *leaf_nodes = 
    malloc(params->expected.len_leaf_nodes * sizeof(struct HODLRLeafNode));
  struct HODLRInternalNode **work_queue = 
    malloc(params->expected.len_work_queue * sizeof(struct HODLRInternalNode *));
  struct HODLRLeafNode **innermost_leaves =
    malloc(params->expected.len_innermost_leaves * sizeof(struct HODLRLeafNode *));
  
  construct_tree(params->height, hodlr, internal_nodes, leaf_nodes, work_queue, 
                 innermost_leaves, &ierr);

  cr_expect(eq(ierr, SUCCESS));
  if (ierr != SUCCESS) {
    free(hodlr); free(internal_nodes); free(leaf_nodes); free(work_queue);
    free(innermost_leaves);
    return;
  }

  expect_tree_consistent(hodlr, params->height, params->expected.len_work_queue);

  cr_log_info("TreeHODLR check concluded, freeing...");
  free_tree_hodlr(&hodlr);
}


ParameterizedTestParameters(allocators, test_allocate_tree_monolithic) {
  int n_params = 6;
  struct ParametersArraySizes *params = generate_array_sizes_params(&n_params);
  return cr_make_param_array(struct ParametersArraySizes, 
                             params, n_params, free_params);
}


ParameterizedTest(struct ParametersArraySizes *params, allocators, 
                  test_allocate_tree_monolithic) {
  cr_log_info("height=%d", params->height);
  int ierr = SUCCESS;

  struct TreeHODLR *hodlr = allocate_tree_monolithic(params->height, &ierr);

  cr_expect(eq(i32, ierr, SUCCESS));
  cr_expect(ne(ptr, hodlr, NULL));
  if (ierr != SUCCESS) {
    return;
  }

  expect_tree_consistent(hodlr, params->height, params->expected.len_work_queue);
  cr_expect(ne(ptr, hodlr->memory_internal_ptr, NULL));
  cr_expect(ne(ptr, hodlr->memory_leaf_ptr, NULL));

  cr_log_info("TreeHODLR check concluded, freeing...");
  free_tree_hodlr(&hodlr);
}


ParameterizedTestParameters(allocators, test_allocate_tree) {
  int n_params = 6;
  struct ParametersArraySizes *params = generate_array_sizes_params(&n_params);
  return cr_make_param_array(struct ParametersArraySizes, 
                             params, n_params, free_params);
}


ParameterizedTest(struct ParametersArraySizes *params, allocators, 
                  test_allocate_tree) {
  cr_log_info("height=%d", params->height);
  int ierr = SUCCESS;

  struct TreeHODLR *hodlr = allocate_tree(params->height, &ierr);

  cr_expect(eq(i32, ierr, SUCCESS));
  cr_expect(ne(ptr, hodlr, NULL));
  if (ierr != SUCCESS) {
    return;
  }

  expect_tree_consistent(hodlr, params->height, params->expected.len_work_queue);
  cr_expect(eq(ptr, hodlr->memory_internal_ptr, NULL));
  cr_expect(eq(ptr, hodlr->memory_leaf_ptr, NULL));

  cr_log_info("TreeHODLR check concluded, freeing...");
  free_tree_hodlr(&hodlr);
}

