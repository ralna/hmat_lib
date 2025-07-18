#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include <math.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../include/utils.h"
#include "../include/common_data.h"

#include "../../src/constructors.c"
#include "../../include/error.h"
#include "../../include/blas_wrapper.h"
#include "../../include/tree.h"


struct ParametersBlockSizes {
  int height;
  int m;
  int *ms;
  int len;
  int *expected;
};


static inline void arrcpy(int *dest, int src[], int len) {
  for (int i = 0; i < len; i++) {
    dest[i] = src[i];
  }
}


void free_block_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersBlockSizes *param = params->params + i;
    cr_free(param->expected);
    cr_free(param->ms);
  }
  cr_free(params->params);
}


static int generate_block_size_params(struct ParametersBlockSizes *params) {
  const int n_params = 8+18;

  int idx = 0, n = 0, tidx;
  int len_heights = 2;
  int heights[] = {1, 2};

  int len_ms = 4;
  int ms[] = {8, 9, 11, 13};

  for (int height_idx = 0; height_idx < len_heights; height_idx++) {
    for (int m_idx = 0; m_idx < len_ms; m_idx++) {
      params[idx].height = heights[height_idx];
      params[idx].m = ms[m_idx];

      n = (int)pow(2, heights[height_idx]) - 1;
      params[idx].len = n;
      params[idx].expected = cr_malloc(n * sizeof(int));
      idx++;
    }
  }

  len_heights = 6;
  int heights2[] = {1, 2, 3, 4, 5, 6};

  len_ms = 3;
  int ms2[] = {139, 597, 2048};

  for (int height_idx = 0; height_idx < len_heights; height_idx++) {
    for (int m_idx = 0; m_idx < len_ms; m_idx++) {
      params[idx].height = heights2[height_idx];
      params[idx].m = ms2[m_idx];
      params[idx].ms = NULL;

      n = (int)pow(2, heights2[height_idx]) - 1;
      params[idx].expected = cr_malloc(n * sizeof(int));
      idx++;
    }
  }

  params[0].expected[0] = 8;
  params[1].expected[0] = 9;
  params[2].expected[0] = 11;
  params[3].expected[0] = 13;
  arrcpy(params[4].expected, (int[]){8, 4, 4}, 3);
  arrcpy(params[5].expected, (int[]){9, 5, 4}, 3);
  arrcpy(params[6].expected, (int[]){11, 6, 5}, 3);
  arrcpy(params[7].expected, (int[]){13, 7, 6}, 3);

  params[8].expected[0] = 139;
  params[9].expected[0] = 597;
  params[10].expected[0] = 2048;
  arrcpy(params[11].expected, (int[]){139, 70, 69}, 3);
  arrcpy(params[12].expected, (int[]){597, 299, 298}, 3);
  arrcpy(params[13].expected, (int[]){2048, 1024, 1024}, 3);
  arrcpy(params[14].expected, (int[]){139, 70, 69, 35, 35, 35, 34}, 7);
  arrcpy(params[15].expected, 
         (int[]){597, 299, 298, 150, 149, 149, 149}, 7);
  arrcpy(params[16].expected, 
         (int[]){2048, 1024, 1024, 512, 512, 512, 512}, 7);
  arrcpy(params[17].expected, 
         (int[]){139, 70, 69, 35, 35, 35, 34, 18, 17, 18, 17, 18, 17, 17, 17}, 
         15);
  arrcpy(params[18].expected, 
         (int[]){597, 299, 298, 150, 149, 149, 149, 
                 75, 75, 75, 74, 75, 74, 75, 74}, 15);
  arrcpy(params[19].expected, 
         (int[]){2048, 1024, 1024, 512, 512, 512, 512,
                 256, 256, 256, 256, 256, 256, 256, 256}, 15);
  arrcpy(params[20].expected, 
        (int[]){139, 70, 69, 35, 35, 35, 34, 18, 17, 18, 17, 18, 17, 17, 17,
                  9, 9, 9, 8, 9, 9, 9, 8, 9, 9, 9, 8, 9, 8, 9, 8}, 31);
  arrcpy(params[21].expected, 
        (int[]){597, 299, 298, 150, 149, 149, 149, 
                75, 75, 75, 74, 75, 74, 75, 74,
                38, 37, 38, 37, 38, 37, 37, 37, 
                38, 37, 37, 37, 38, 37, 37, 37}, 31);
  arrcpy(params[22].expected, 
        (int[]){2048, 1024, 1024, 512, 512, 512, 512,
                  256, 256, 256, 256, 256, 256, 256, 256,
                  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                  128, 128, 128, 128}, 31);
  arrcpy(params[23].expected, 
        (int[]){139, 70, 69, 35, 35, 35, 34, 18, 17, 18, 17, 18, 17, 17, 17,
                  9, 9, 9, 8, 9, 9, 9, 8, 9, 9, 9, 8, 9, 8, 9, 8,
                  5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 5, 4,
                  5, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4}, 
        63);
  arrcpy(params[24].expected, 
        (int[]){597, 299, 298, 150, 149, 149, 149, 
                75, 75, 75, 74, 75, 74, 75, 74,
                38, 37, 38, 37, 38, 37, 37, 37, 
                38, 37, 37, 37, 38, 37, 37, 37,
                19, 19, 19, 18, 19, 19, 19, 18,
                19, 19, 19, 18, 19, 18, 19, 18,
                19, 19, 19, 18, 19, 18, 19, 18,
                19, 19, 19, 18, 19, 18, 19, 18}, 63);
  arrcpy(params[25].expected, 
        (int[]){2048, 1024, 1024, 512, 512, 512, 512,
                  256, 256, 256, 256, 256, 256, 256, 256,
                  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                  128, 128, 128, 128,
                  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                  64, 64}, 63);

  return n_params;
}


ParameterizedTestParameters(constructors, block_sizes_halves) {
  static int n_params = 26; 
  struct ParametersBlockSizes *params = 
    cr_malloc(n_params * sizeof(struct ParametersBlockSizes));

  int actual = generate_block_size_params(params);

  if (actual != n_params) {
    printf("INCORRECT PARAMETER ALLOCATION - allocated %d but set %d",
           n_params, actual);
  }

  return cr_make_param_array(struct ParametersBlockSizes, params, n_params, 
                             free_block_params);
}


ParameterizedTest(struct ParametersBlockSizes *params, constructors,
                  block_sizes_halves) {
  cr_log_info("height=%d, m=%d", params->height, params->m);

  int ierr = SUCCESS;
  
  struct TreeHODLR *hodlr = 
    allocate_tree_monolithic(params->height, &ierr, &malloc, &free);
  if (hodlr == NULL) {
    cr_fail("Allocation failed");
    return;
  }

  struct HODLRInternalNode **queue = 
    compute_block_sizes_halves(hodlr, params->m);

  long q_next_node_density = hodlr->len_work_queue;
  long q_current_node_density = q_next_node_density;
  int qidx = 0, eidx = 1, len_queue = 1;

  queue[0] = hodlr->root;
  cr_expect(eq(int, queue[0]->m, params->expected[0]),
            "@depth=%d & node=%d (idx=%d)", 0, 0, 0);

  for (int height = 1; height < params->height; height++) {
    q_next_node_density /= 2;
    for (int parent = 0; parent < len_queue; parent++) {
      qidx = parent * q_current_node_density;
      cr_expect(eq(int, queue[qidx]->children[0].internal->m, 
                   params->expected[eidx]),
                "@depth=%d & node=%d (idx=%d)", height, qidx, eidx);
      eidx++;
      cr_expect(eq(int, queue[qidx]->children[3].internal->m, 
                   params->expected[eidx]),
                "@depth=%d & node=%d (idx=%d)", height, qidx, eidx);
      eidx++;

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[qidx]->children[3].internal;
      queue[qidx] = queue[qidx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  free_tree_hodlr(&hodlr, &free);
}


ParameterizedTestParameters(constructors, block_sizes_custom) {
  static int n_params = 26; 
  struct ParametersBlockSizes *params = 
    cr_malloc(n_params * sizeof(struct ParametersBlockSizes));

  int actual = generate_block_size_params(params);
  
  for (int i = 0; i < n_params; i++) {
    params[i].ms = cr_malloc((int)pow(2, params[i].height) * sizeof(int));
  }
  arrcpy(params[0].ms, (int[]){4, 4}, 2);
  arrcpy(params[1].ms, (int[]){5, 4}, 2);
  arrcpy(params[2].ms, (int[]){6, 5}, 2);
  arrcpy(params[3].ms, (int[]){7, 6}, 2);
  arrcpy(params[4].ms, (int[]){2, 2, 2, 2}, 4);
  arrcpy(params[5].ms, (int[]){3, 2, 2, 2}, 4);
  arrcpy(params[6].ms, (int[]){3, 3, 3, 2}, 4);
  arrcpy(params[7].ms, (int[]){4, 3, 3, 3}, 4);

  arrcpy(params[8].ms, (int[]){70, 69}, 2);
  arrcpy(params[9].ms, (int[]){299, 298}, 2);
  arrcpy(params[10].ms, (int[]){1024, 1024}, 2);
  arrcpy(params[11].ms, (int[]){35, 35, 35, 34}, 4);
  arrcpy(params[12].ms, (int[]){150, 149, 149, 149}, 4);
  arrcpy(params[13].ms, (int[]){512, 512, 512, 512}, 4);
  arrcpy(params[14].ms, (int[]){18, 17, 18, 17, 18, 17, 17, 17}, 8);
  arrcpy(params[15].ms, (int[]){75, 75, 75, 74, 75, 74, 75, 74}, 8);
  arrcpy(params[16].ms, 
         (int[]){256, 256, 256, 256, 256, 256, 256, 256}, 8);
  arrcpy(params[17].ms, 
         (int[]){9, 9, 9, 8, 
                 9, 9, 9, 8, 
                 9, 9, 9, 8, 
                 9, 8, 9, 8}, 16);
  arrcpy(params[18].ms, 
         (int[]){38, 37, 38, 37, 
                 38, 37, 37, 37, 
                 38, 37, 37, 37, 
                 38, 37, 37, 37}, 16);
  arrcpy(params[19].ms, 
         (int[]){128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 
                 128, 128, 128, 128}, 16);
  arrcpy(params[20].ms, 
         (int[]){5, 4, 5, 4, 5, 4, 4, 4, 
                 5, 4, 5, 4, 5, 4, 4, 4, 
                 5, 4, 5, 4, 5, 4, 4, 4, 
                 5, 4, 4, 4, 5, 4, 4, 4}, 32);
  arrcpy(params[21].ms, 
         (int[]){19, 19, 19, 18, 19, 19, 19, 18,
                 19, 19, 19, 18, 19, 18, 19, 18,
                 19, 19, 19, 18, 19, 18, 19, 18,
                 19, 19, 19, 18, 19, 18, 19, 18}, 32);
  arrcpy(params[22].ms, 
         (int[]){64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                 64, 64}, 32);
  arrcpy(params[23].ms, 
         (int[]){3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 
                 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 
                 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2}, 64);
  arrcpy(params[24].ms, 
         (int[]){10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 10, 9, 10, 9, 9, 9,
                 10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9,
                 10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9,
                 10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9}, 64);
  arrcpy(params[25].ms, 
         (int[]){32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32},
         64);

  if (actual != n_params) {
    printf("INCORRECT PARAMETER ALLOCATION - allocated %d but set %d",
           n_params, actual);
  }

  return cr_make_param_array(struct ParametersBlockSizes, params, n_params, 
                             free_block_params);
}


ParameterizedTest(struct ParametersBlockSizes *params, constructors,
                  block_sizes_custom) {
  cr_log_info("height=%d, m=%d", params->height, params->m);

  int ierr = SUCCESS;
  
  struct TreeHODLR *hodlr = 
    allocate_tree_monolithic(params->height, &ierr, &malloc, &free);
  if (hodlr == NULL) {
    cr_fail("Allocation failed");
    return;
  }

  struct HODLRInternalNode **queue = 
    compute_block_sizes_custom(hodlr, params->ms);

  long q_next_node_density = hodlr->len_work_queue;
  long q_current_node_density = q_next_node_density;
  int qidx = 0, eidx = 1, len_queue = 1;

  queue[0] = hodlr->root;
  cr_expect(eq(int, queue[0]->m, params->expected[0]),
            "@depth=%d & node=%d (idx=%d)", 0, 0, 0);

  for (int height = 1; height < params->height; height++) {
    q_next_node_density /= 2;
    for (int parent = 0; parent < len_queue; parent++) {
      qidx = parent * q_current_node_density;
      cr_expect(eq(int, queue[qidx]->children[0].internal->m, 
                   params->expected[eidx]),
                "@depth=%d & node=%d (idx=%d)", height, qidx, eidx);
      eidx++;
      cr_expect(eq(int, queue[qidx]->children[3].internal->m, 
                   params->expected[eidx]),
                "@depth=%d & node=%d (idx=%d)", height, qidx, eidx);
      eidx++;

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[qidx]->children[3].internal;
      queue[qidx] = queue[qidx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  free_tree_hodlr(&hodlr, &free);
}


struct ParametersCopyDiag {
  int m;
  double *matrix;
  long len;
  double **expected_data;
  int *expected_m;
};


void free_copy_diag_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersCopyDiag *param = 
        (struct ParametersCopyDiag *)params->params + i;
    cr_free(param->matrix);
    for (int j = 0; j < param->len; j++) {
      cr_free(param->expected_data[j]);
    }
    cr_free(param->expected_data);
    cr_free(param->expected_m);
  }
  cr_free(params->params);
}


static double * construct_diagonal_increasing(const int m, double start) {
  int idx;
  double *matrix = cr_malloc(m * m * sizeof(double));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      idx = j + i * m;
      if (i == j) {
        matrix[idx] = start;
        start++;
      } else {
        matrix[idx] = 0;
      }
    }
  }

  return matrix;
}


static int fill_copy_diagonal_const(struct ParametersCopyDiag *params,
                                    double *(*func)(int)) {
  const int len_ms = 3;
  const int ms[] = {21, 33, 64};

  const int len_lens = 3;
  const int lens[] = {2, 4, 8};

  int idx = 0, m = 0, len = 0, m_smaller = 0, first_two = 0;
  for (int midx = 0; midx < len_ms; midx++) {
    for (int lidx = 0; lidx < len_lens; lidx++) {
      m = ms[midx];
      len = lens[lidx];

      params[idx].m = m;
      params[idx].len = len;
      params[idx].matrix = func(m);

      m_smaller = m / len;
      params[idx].expected_m = cr_malloc(len * sizeof(int));
      first_two = 2 * m_smaller + (m - len * m_smaller);
      params[idx].expected_m[1] = first_two / 2;
      params[idx].expected_m[0] = first_two - params[idx].expected_m[1];
      for (int i = 2; i < len; i++) {
        params[idx].expected_m[i] = m_smaller;
      }

      params[idx].expected_data = cr_malloc(len * sizeof(double *));
      for (int i = 0; i < len; i++) {
        params[idx].expected_data[i] = func(params[idx].expected_m[i]);
      }
      idx++;
    }
  }
  return len_ms * len_lens;
}


static int fill_copy_diagonal_var(struct ParametersCopyDiag *params,
                                  double *(*func)(int, double)) {
  const int len_ms = 3;
  const int ms[] = {21, 33, 64};

  const int len_lens = 3;
  const int lens[] = {2, 4, 8};

  int idx = 0, m = 0, len = 0, m_smaller = 0, first_two = 0;
  double start = 0.0;

  for (int midx = 0; midx < len_ms; midx++) {
    for (int lidx = 0; lidx < len_lens; lidx++) {
      start = 0.0;
      m = ms[midx];
      len = lens[lidx];

      params[idx].m = m;
      params[idx].len = len;
      params[idx].matrix = func(m, 0.0);

      m_smaller = m / len;
      params[idx].expected_m = cr_malloc(len * sizeof(int));
      first_two = 2 * m_smaller + (m - len * m_smaller);
      params[idx].expected_m[1] = first_two / 2;
      params[idx].expected_m[0] = first_two - params[idx].expected_m[1];
      for (int i = 2; i < len; i++) {
        params[idx].expected_m[i] = m_smaller;
      }

      params[idx].expected_data = cr_malloc(len * sizeof(double *));
      for (int i = 0; i < len; i++) {
        params[idx].expected_data[i] = func(params[idx].expected_m[i], start);
        start += params[idx].expected_m[i];
      }
      idx++;
    }
  }
  return len_ms * len_lens;
}


ParameterizedTestParameters(constructors, copy_diagonal) {
  const int n_params = 2*9; int actual_n_params = 0;
  struct ParametersCopyDiag *params = 
    cr_malloc(n_params * sizeof(struct ParametersCopyDiag));

  actual_n_params += fill_copy_diagonal_const(params, &construct_laplacian_matrix);
  actual_n_params += fill_copy_diagonal_var(params + actual_n_params, 
                                            &construct_diagonal_increasing);

  if (n_params != actual_n_params) {
    printf("PARAMETER SETUP FAILED\n");
  }
  return cr_make_param_array(struct ParametersCopyDiag, params, n_params, 
                             free_copy_diag_params);
}


ParameterizedTest(struct ParametersCopyDiag *params, constructors, 
                  copy_diagonal) {
  cr_log_info("m=%d, len=%ld", params->m, params->len);

  // Set up input parameters
  long n_parent_nodes = params->len / 2;
  struct HODLRInternalNode **queue = 
    malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (int i = 0; i < n_parent_nodes; i++) {
    queue[i] = malloc(sizeof(struct HODLRInternalNode));
    queue[i]->children[0].leaf = malloc(sizeof(struct HODLRLeafNode));
    queue[i]->children[3].leaf = malloc(sizeof(struct HODLRLeafNode));
    queue[i]->m = params->expected_m[2 * i] + params->expected_m[2 * i + 1];
  }

  int ierr = SUCCESS, idx = 0;
  copy_diagonal_blocks(params->matrix, params->m, queue, n_parent_nodes, &ierr,
                       &malloc);

  cr_expect(eq(int, ierr, SUCCESS));
  
  for (int i = 0; i < n_parent_nodes; i++) {
    if (queue[i]->children[0].leaf->data.diagonal.data == NULL) {
      cr_fail("Data matrix for node %d (parent=%d, leaf=0) is NULL", idx, i);
      break;
    }
    
    if (queue[i]->children[0].leaf->data.diagonal.m != params->expected_m[idx]) {
      cr_fail("Node %d (parent=%d, leaf=0) has different dimensions - "
              "expected=%d, actual=%d", idx, i, 
              queue[i]->children[0].leaf->data.diagonal.m, 
              params->expected_m[idx]);
    } else {
      expect_matrix_double_eq(
        queue[i]->children[0].leaf->data.diagonal.data,
        params->expected_data[idx],
        queue[i]->children[0].leaf->data.diagonal.m,
        queue[i]->children[0].leaf->data.diagonal.m,
        queue[i]->children[0].leaf->data.diagonal.m,
        params->expected_m[idx],
        idx
      );
    }
    idx++;

    if (queue[i]->children[3].leaf->data.diagonal.data == NULL) {
      cr_fail("Data matrix for node %d (parent=%d, leaf=3) is NULL", idx, i);
      break;
    }
    if (queue[i]->children[3].leaf->data.diagonal.m != params->expected_m[idx]) {
      cr_fail("Node %d (parent=%d, leaf=0) has different dimensions - "
              "expected=%d, actual=%d", idx, i, 
              queue[i]->children[3].leaf->data.diagonal.m, 
              params->expected_m[idx]);
    } else {
      expect_matrix_double_eq(
        queue[i]->children[3].leaf->data.diagonal.data,
        params->expected_data[idx],
        queue[i]->children[3].leaf->data.diagonal.m,
        queue[i]->children[3].leaf->data.diagonal.m,
        queue[i]->children[3].leaf->data.diagonal.m,
        params->expected_m[idx],
        idx
      );
    }
    idx++;
  }

  // Free 
  for (int i = 0; i < n_parent_nodes; i++) {
    for (int leaf = 0; leaf < 4; leaf+=3) {
      free(queue[i]->children[leaf].leaf->data.diagonal.data);
      free(queue[i]->children[leaf].leaf);
    }
    free(queue[i]);
  }
  free(queue);
}


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
  for (size_t i = 0; i < params->length; i++) {
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
  int ierr = SUCCESS;
  int n_singular_values = params->m < params->n ? params->m : params->n;
  
  double *s_work = malloc(n_singular_values * sizeof(double));
  double *u_work = malloc(params->m * n_singular_values * sizeof(double));
  double *vt_work = malloc(params->n * n_singular_values * sizeof(double));

  int result_code = compress_off_diagonal(
      &result, params->m, params->n, n_singular_values, params->m_full, 
      params->matrix, s_work, u_work, vt_work, params->svd_threshold, &ierr,
      &malloc
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

  expect_matrix_double_eq(result.u, params->u_expected, params->m, params->expected_n_singular,
                       result.m, params->m, 'U');
  expect_matrix_double_eq(result.v, params->v_expected, params->n, params->expected_n_singular,
                       result.m, params->m, 'V');
}


ParameterizedTestParameters(tree, recompress) {
  int n_params;
  struct ParametersTestCompress *params = generate_compress_params(&n_params);
  return cr_make_param_array(struct ParametersTestCompress, params, n_params, free_compress_params);
}


ParameterizedTest(struct ParametersTestCompress *params, tree, recompress) {
  struct NodeOffDiagonal node;
  int ierr = SUCCESS;
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
      &ierr, &malloc
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

  expect_matrix_double_eq(result, og_matrix, node.m, node.n, 
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


void free_dense_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersTestDense *param = 
      (struct ParametersTestDense *)params->params + i;
    cr_free(param->matrix);
    free_tree_hodlr(&param->expected, &cr_free);
  }
  cr_free(params->params);
}


static inline void set_up_off_diagonal(struct NodeOffDiagonal *node,
                                       const int m,
                                       const int n,
                                       const int s) {
  node->m = m; node->n = n; node->s = s;
  node->u = cr_calloc(m * s, sizeof(double));
  node->v = cr_calloc(n * s, sizeof(double));
}


struct ParametersTestDense * generate_dense_params(int * len) {
  const int n_params = 5; int ierr = SUCCESS;
  *len = n_params;
  struct ParametersTestDense *params = cr_malloc(n_params * sizeof(struct ParametersTestDense));
  
  int idx = 0;
  const int m = 69; int m_larger, m_smaller;
  for (int height = 1; height < 6; height++) {
    params[idx].height = height;
    params[idx].m = m;
    params[idx].matrix = construct_laplacian_matrix(m);
    params[idx].matrix[m - 1] = 0.5;
    params[idx].matrix[m * (m - 1)] = 0.5;
    params[idx].svd_threshold = 1e-8;
    params[idx].expected = 
      allocate_tree_monolithic(height, &ierr, &cr_malloc, &cr_free);
    compute_block_sizes_halves(params[idx].expected, m);

    struct HODLRInternalNode **queue = params[idx].expected->work_queue;
    int n_parent_nodes = params[idx].expected->len_work_queue;

    // Diagonal blocks
    for (int parent = 0; parent < n_parent_nodes; parent++) {
      queue[parent] = params[idx].expected->innermost_leaves[2 * parent]->parent;
      
      m_smaller = queue[parent]->m / 2;
      m_larger = queue[parent]->m - m_smaller;

      params[idx].expected->innermost_leaves[2 * parent]->data.diagonal.m = m_larger;
      params[idx].expected->innermost_leaves[2 * parent]->data.diagonal.data 
        = construct_laplacian_matrix(m_larger);

      params[idx].expected->innermost_leaves[2 * parent + 1]->data.diagonal.m 
        = m_smaller;
      params[idx].expected->innermost_leaves[2 * parent + 1]->data.diagonal.data 
        = construct_laplacian_matrix(m_smaller);   
    }

    for (int level = height; level > 1; level--) {
      for (int node = 0; node < n_parent_nodes; node++) {
        m_smaller = queue[node]->m / 2;
        m_larger = queue[node]->m - m_smaller;
        
        set_up_off_diagonal(&queue[node]->children[1].leaf->data.off_diagonal, 
                            m_larger, m_smaller, 1);
        queue[node]->children[1].leaf->data.off_diagonal.u[m_larger-1] = 1.0;
        queue[node]->children[1].leaf->data.off_diagonal.v[0] = -1.0;
  
        set_up_off_diagonal(&queue[node]->children[2].leaf->data.off_diagonal, 
                            m_smaller, m_larger, 1);
        queue[node]->children[2].leaf->data.off_diagonal.u[0] = 1.0;
        queue[node]->children[2].leaf->data.off_diagonal.v[m_larger-1] = -1.0;

        queue[node / 2] = queue[node]->parent;
      }
      n_parent_nodes /= 2;
    }

    m_smaller = m / 2; m_larger = m - m_smaller;
    set_up_off_diagonal(
      &queue[0]->children[1].leaf->data.off_diagonal,
      m_larger, m_smaller, 2    
    );
    queue[0]->children[1].leaf->data.off_diagonal.u[m_larger-1] = 1.0;
    queue[0]->children[1].leaf->data.off_diagonal.u[m_larger] = 0.5;
    queue[0]->children[1].leaf->data.off_diagonal.v[0] = -1.0;
    queue[0]->children[1].leaf->data.off_diagonal.v[2 * m_smaller - 1] = 1.0;

    set_up_off_diagonal(
      &queue[0]->children[2].leaf->data.off_diagonal,
      m_smaller, m_larger, 2
    );
    queue[0]->children[2].leaf->data.off_diagonal.u[0] = 1.0;
    queue[0]->children[2].leaf->data.off_diagonal.u[2 * m_smaller - 1] = -0.5;
    queue[0]->children[2].leaf->data.off_diagonal.v[m_larger-1] = -1.0;
    queue[0]->children[2].leaf->data.off_diagonal.v[m_larger] = -1.0;

    idx++;
  }
  
  return params;
}


ParameterizedTestParameters(constructors, dense_to_tree) {
  int n_params;
  struct ParametersTestDense *params = generate_dense_params(&n_params);
  return cr_make_param_array(struct ParametersTestDense, params, n_params, free_dense_params);
}


ParameterizedTest(struct ParametersTestDense *params, 
                  constructors, dense_to_tree) {
  int ierr;
  cr_log_info("height=%d, m=%d", params->height, params->m);

  struct TreeHODLR *result = allocate_tree_monolithic(params->height, &ierr,
                                                      &malloc, &free);
  cr_expect(eq(ierr, SUCCESS));
  cr_expect(ne(result, NULL));
  if (ierr != SUCCESS) {
    cr_fatal("Tree HODLR allocation failed");
  }

  int svd = dense_to_tree_hodlr(
    result, params->m, NULL, params->matrix, params->svd_threshold, &ierr, 
    &malloc, &free
  );
  
  cr_expect(eq(ierr, SUCCESS));
  cr_expect(zero(svd));
  if (ierr != SUCCESS) {
    free_tree_hodlr(&result, &free);
    cr_fatal();
  }

  expect_tree_hodlr(result, params->expected);
  free_tree_hodlr(&result, &free);
}
