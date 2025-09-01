#ifndef _TEST_HODLR
#define _TEST_HODLR 1
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include "../utils/utils.h"
#include "../utils/common_data.h"

#include "../../src/constructors.c"
#include "../../include/hmat_lib/error.h"
#include "../../include/hmat_lib/hodlr.h"
#include "../../include/hmat_lib/allocators.h"
#include "../../include/internal/blas_wrapper.h"


#define STR_LEN 10


static inline void check_n_params(const int actual, const int allocd) {
  if (actual != allocd) {
    cr_log_error("INCORRECT PARAMETER ALLOCATION - allocated %d but set %d\n",
                 allocd, actual);
  }
}


#define CREATE_SETUP_FUNC(STRUCT, FIELD) \
static inline void FIELD##_setup( \
  struct STRUCT *params, \
  int *idx, \
  int len, \
  int src[] \
) { \
  params[*idx].FIELD = cr_malloc(len * sizeof(int)); \
  for (int i = 0; i < len; i++) params[*idx].FIELD[i] = src[i]; \
  (*idx)++; \
}


static inline void expect_block_sizes(
  const int m_larger,
  const int m_smaller,
  struct HODLRInternalNode **queue,
  const int level,
  const int node,
  const int eidx,
  const bool internal
) {
  if (internal == true) {
    cr_expect(
      eq(int, queue[node]->children[0].internal->m, m_larger),
      "internal @level=%d & node=%d (idx=%d)", level, node, eidx-2
    );
    cr_expect(
      eq(int, queue[node]->children[3].internal->m, m_smaller),
      "internal @level=%d & node=%d (idx=%d)", level, node, eidx-1
    );
  } else {
    cr_expect(
      eq(int, queue[node]->children[0].leaf->data.diagonal.m, m_larger),
      "diagonal @level=%d & node=%d (idx=%d)", level, node, eidx-2
    );
    cr_expect(
      eq(int, queue[node]->children[3].leaf->data.diagonal.m, m_smaller),
      "diagonal @level=%d & node=%d (idx=%d)", level, node, eidx-1
    );
  }

  cr_expect(
    eq(int, queue[node]->children[1].leaf->data.off_diagonal.m, m_larger),
    "off-diagonal @level=%d & node=%d (idx=%d)", level, node, eidx-2
  );
  cr_expect(
    eq(int, queue[node]->children[1].leaf->data.off_diagonal.n, m_smaller),
    "off-diagonal @level=%d & node=%d (idx=%d)", level, node, eidx-1
  );

  cr_expect(
    eq(int, queue[node]->children[2].leaf->data.off_diagonal.m, m_smaller),
    "off-diagonal @level=%d & node=%d (idx=%d)", level, node, eidx-1
  );
  cr_expect(
    eq(int, queue[node]->children[2].leaf->data.off_diagonal.n, m_larger),
    "off-diagonal @level=%d & node=%d (idx=%d)", level, node, eidx-2
  );
}


struct ParametersBlockSizes {
  int height;
  int *ms;
  int *expected;
};


CREATE_SETUP_FUNC(ParametersBlockSizes, expected)
CREATE_SETUP_FUNC(ParametersBlockSizes, ms)


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

  int idx = 0;
  enum {len_heights1 = 2};
  const int heights[len_heights1] = {1, 2};

  const int len_ms1 = 4;
  for (int height_idx = 0; height_idx < len_heights1; height_idx++) {
    for (int m_idx = 0; m_idx < len_ms1; m_idx++) {
      params[idx].height = heights[height_idx];
      params[idx].ms = NULL;
      idx++;
    }
  }

  enum {len_heights2 = 6};
  const int heights2[len_heights2] = {1, 2, 3, 4, 5, 6};

  const int len_ms2 = 3;
  for (int height_idx = 0; height_idx < len_heights2; height_idx++) {
    for (int m_idx = 0; m_idx < len_ms2; m_idx++) {
      params[idx].height = heights2[height_idx];
      params[idx].ms = NULL;
      idx++;
    }
  }

  check_n_params(idx, n_params);
  idx = 0;

  expected_setup(params, &idx, 3, (int[]){8, 4, 4});
  expected_setup(params, &idx, 3, (int[]){9, 5, 4});
  expected_setup(params, &idx, 3, (int[]){11, 6, 5});
  expected_setup(params, &idx, 3, (int[]){13, 7, 6});

  expected_setup(params, &idx, 7, (int[]){8, 4, 4, 2, 2, 2, 2});
  expected_setup(params, &idx, 7, (int[]){9, 5, 4, 3, 2, 2, 2});
  expected_setup(params, &idx, 7, (int[]){11, 6, 5, 3, 3, 3, 2});
  expected_setup(params, &idx, 7, (int[]){13, 7, 6, 4, 3, 3, 3});

  expected_setup(params, &idx, 3, (int[]){139, 70, 69});
  expected_setup(params, &idx, 3, (int[]){597, 299, 298});
  expected_setup(params, &idx, 3, (int[]){2048, 1024, 1024});
  expected_setup(params, &idx, 7, (int[]){139, 70, 69, 35, 35, 35, 34});
  expected_setup(params, &idx, 7, 
                 (int[]){597, 299, 298, 150, 149, 149, 149});
  expected_setup(params, &idx, 7, 
                 (int[]){2048, 1024, 1024, 512, 512, 512, 512});
  expected_setup(
    params, &idx, 15, 
    (int[]){139, 70, 69, 35, 35, 35, 34, 18, 17, 18, 17, 18, 17, 17, 17}
  );
  expected_setup(params, &idx, 15, 
                 (int[]){597, 299, 298, 150, 149, 149, 149, 
                         75, 75, 75, 74, 75, 74, 75, 74});
  expected_setup(params, &idx, 15, 
                 (int[]){2048, 1024, 1024, 512, 512, 512, 512,
                         256, 256, 256, 256, 256, 256, 256, 256});
  expected_setup(params, &idx, 31, 
                 (int[]){139, 70, 69, 35, 35, 35, 34, 
                         18, 17, 18, 17, 18, 17, 17, 17,
                         9, 9, 9, 8, 9, 9, 9, 8, 9, 9, 9, 8, 9, 8, 9, 8});
  expected_setup(params, &idx, 31, 
                 (int[]){597, 299, 298, 150, 149, 149, 149, 
                         75, 75, 75, 74, 75, 74, 75, 74,
                         38, 37, 38, 37, 38, 37, 37, 37, 
                         38, 37, 37, 37, 38, 37, 37, 37});
  expected_setup(params, &idx, 31, 
                 (int[]){2048, 1024, 1024, 512, 512, 512, 512,
                         256, 256, 256, 256, 256, 256, 256, 256,
                         128, 128, 128, 128, 128, 128, 128, 128, 
                         128, 128, 128, 128, 128, 128, 128, 128});
  expected_setup(params, &idx, 63, 
                 (int[]){139, 70, 69, 35, 35, 35, 34, 
                         18, 17, 18, 17, 18, 17, 17, 17,
                         9, 9, 9, 8, 9, 9, 9, 8, 9, 9, 9, 8, 9, 8, 9, 8,
                         5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 5, 4, 5, 4, 4, 4, 
                         5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4});
  expected_setup(params, &idx, 63, 
                 (int[]){597, 299, 298, 150, 149, 149, 149, 
                         75, 75, 75, 74, 75, 74, 75, 74,
                         38, 37, 38, 37, 38, 37, 37, 37, 
                         38, 37, 37, 37, 38, 37, 37, 37,
                         19, 19, 19, 18, 19, 19, 19, 18,
                         19, 19, 19, 18, 19, 18, 19, 18,
                         19, 19, 19, 18, 19, 18, 19, 18,
                         19, 19, 19, 18, 19, 18, 19, 18});
  expected_setup(params, &idx, 63, 
                 (int[]){2048, 1024, 1024, 512, 512, 512, 512,
                         256, 256, 256, 256, 256, 256, 256, 256,
                         128, 128, 128, 128, 128, 128, 128, 128, 
                         128, 128, 128, 128, 128, 128, 128, 128,
                         64, 64, 64, 64, 64, 64, 64, 64, 
                         64, 64, 64, 64, 64, 64, 64, 64, 
                         64, 64, 64, 64, 64, 64, 64, 64, 
                         64, 64, 64, 64, 64, 64, 64, 64});
  expected_setup(params, &idx, 127, 
                 (int[]){139, 70, 69, 35, 35, 35, 34, 
                         18, 17, 18, 17, 18, 17, 17, 17,
                         9, 9, 9, 8, 9, 9, 9, 8, 9, 9, 9, 8, 9, 8, 9, 8,
                         5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 5, 4, 5, 4, 4, 4, 
                         5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4,
                         3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                         3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                         3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                         3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2});
  expected_setup(params, &idx, 127, 
                 (int[]){597, 299, 298, 150, 149, 149, 149, 
                         75, 75, 75, 74, 75, 74, 75, 74,
                         38, 37, 38, 37, 38, 37, 37, 37, 
                         38, 37, 37, 37, 38, 37, 37, 37,
                         19, 19, 19, 18, 19, 19, 19, 18,
                         19, 19, 19, 18, 19, 18, 19, 18,
                         19, 19, 19, 18, 19, 18, 19, 18,
                         19, 19, 19, 18, 19, 18, 19, 18,
                         10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 10, 9, 10, 9, 9, 9,
                         10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9,
                         10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9,
                         10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9});
  expected_setup(params, &idx, 127, 
                 (int[]){2048, 1024, 1024, 512, 512, 512, 512,
                         256, 256, 256, 256, 256, 256, 256, 256,
                         128, 128, 128, 128, 128, 128, 128, 128, 
                         128, 128, 128, 128, 128, 128, 128, 128,
                         64, 64, 64, 64, 64, 64, 64, 64, 
                         64, 64, 64, 64, 64, 64, 64, 64, 
                         64, 64, 64, 64, 64, 64, 64, 64, 
                         64, 64, 64, 64, 64, 64, 64, 64,
                         32, 32, 32, 32, 32, 32, 32, 32,
                         32, 32, 32, 32, 32, 32, 32, 32,
                         32, 32, 32, 32, 32, 32, 32, 32,
                         32, 32, 32, 32, 32, 32, 32, 32,
                         32, 32, 32, 32, 32, 32, 32, 32,
                         32, 32, 32, 32, 32, 32, 32, 32,
                         32, 32, 32, 32, 32, 32, 32, 32,
                         32, 32, 32, 32, 32, 32, 32, 32});

  check_n_params(idx, n_params);

  return n_params;
}


ParameterizedTestParameters(constructors, block_sizes_halves) {
  const int n_params = 26; 
  struct ParametersBlockSizes *params = 
    cr_malloc(n_params * sizeof(struct ParametersBlockSizes));

  int actual = generate_block_size_params(params);

  check_n_params(actual, n_params);

  return cr_make_param_array(struct ParametersBlockSizes, params, n_params, 
                             free_block_params);
}


ParameterizedTest(struct ParametersBlockSizes *params, constructors,
                  block_sizes_halves) {
  cr_log_info("height=%d, m=%d", params->height, params->expected[0]);

  int ierr = SUCCESS;
  
  struct TreeHODLR *hodlr = 
    allocate_tree_monolithic(params->height, &ierr, &malloc, &free);
  if (hodlr == NULL) {
    cr_fail("Allocation failed");
    return;
  }

  struct HODLRInternalNode **queue = 
    compute_block_sizes_halves(hodlr, params->expected[0]);

  for (int parent = 0; parent < hodlr->len_work_queue; parent++) {
    cr_expect(eq(ptr, queue[parent], 
                 hodlr->innermost_leaves[2 * parent]->parent));
  }

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
      expect_block_sizes(params->expected[eidx], params->expected[eidx+1], 
                         queue, height, qidx, eidx+2, true);
      eidx += 2;

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[qidx]->children[3].internal;
      queue[qidx] = queue[qidx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  for (int parent = 0; parent < len_queue; parent++) {
    const int m_larger = params->expected[eidx]; eidx++;
    const int m_smaller = params->expected[eidx]; eidx++;
    expect_block_sizes(m_larger, m_smaller, queue, -1, parent, eidx, false);
  }

  free_tree_hodlr(&hodlr, &free);
}


static inline int generate_block_size_params_uneven(
  struct ParametersBlockSizes *params
) {
  enum {n_params = 3};
  const int heights[n_params] = {2, 2, 3};

  for (int i = 0; i < n_params; i++) {
    params[i].height = heights[i];
  }

  int idx1 = 0, idx2 = 0;
  ms_setup(params, &idx1, 4, (int[]){3264, 3264, 3261, 3255});
  expected_setup(params, &idx2, 7, 
                 (int[]){13044, 6528, 6516, 3264, 3264, 3261, 3255});

  ms_setup(params, &idx1, 4, (int[]){5000, 500, 1, 4499});
  expected_setup(params, &idx2, 7, 
                 (int[]){10000, 5500, 4500, 5000, 500, 1, 4499});

  ms_setup(params, &idx1, 8,
           (int[]){10, 11, 1, 9, 5, 20, 7, 6});
  expected_setup(params, &idx2, 15,
                 (int[]){69, 31, 38, 21, 10, 25, 13, 10, 11, 1, 9, 5, 20, 7, 6});

  check_n_params(idx1, n_params); check_n_params(idx2, n_params);

  return n_params;
}


ParameterizedTestParameters(constructors, block_sizes_custom) {
  const int n_params = 26+3; 
  struct ParametersBlockSizes *params = 
    cr_malloc(n_params * sizeof(struct ParametersBlockSizes));

  int actual = generate_block_size_params(params);
  
  int idx = 0;
  ms_setup(params, &idx, 2, (int[]){4, 4});
  ms_setup(params, &idx, 2, (int[]){5, 4});
  ms_setup(params, &idx, 2, (int[]){6, 5});
  ms_setup(params, &idx, 2, (int[]){7, 6});
  ms_setup(params, &idx, 4, (int[]){2, 2, 2, 2});
  ms_setup(params, &idx, 4, (int[]){3, 2, 2, 2});
  ms_setup(params, &idx, 4, (int[]){3, 3, 3, 2});
  ms_setup(params, &idx, 4, (int[]){4, 3, 3, 3});

  ms_setup(params, &idx, 2, (int[]){70, 69});
  ms_setup(params, &idx, 2, (int[]){299, 298});
  ms_setup(params, &idx, 2, (int[]){1024, 1024});
  ms_setup(params, &idx, 4, (int[]){35, 35, 35, 34});
  ms_setup(params, &idx, 4, (int[]){150, 149, 149, 149});
  ms_setup(params, &idx, 4, (int[]){512, 512, 512, 512});
  ms_setup(params, &idx, 8, (int[]){18, 17, 18, 17, 18, 17, 17, 17});
  ms_setup(params, &idx, 8, (int[]){75, 75, 75, 74, 75, 74, 75, 74});
  ms_setup(params, &idx, 8, (int[]){256, 256, 256, 256, 256, 256, 256, 256});
  ms_setup(params, &idx, 16, 
           (int[]){9, 9, 9, 8, 
                   9, 9, 9, 8, 
                   9, 9, 9, 8, 
                   9, 8, 9, 8});
  ms_setup(params, &idx, 16, 
           (int[]){38, 37, 38, 37, 
                   38, 37, 37, 37, 
                   38, 37, 37, 37, 
                   38, 37, 37, 37});
  ms_setup(params, &idx, 16, 
          (int[]){128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 
                  128, 128, 128, 128});
  ms_setup(params, &idx, 32, 
           (int[]){5, 4, 5, 4, 5, 4, 4, 4, 
                   5, 4, 5, 4, 5, 4, 4, 4, 
                   5, 4, 5, 4, 5, 4, 4, 4, 
                   5, 4, 4, 4, 5, 4, 4, 4});
  ms_setup(params, &idx, 32, 
           (int[]){19, 19, 19, 18, 19, 19, 19, 18,
                   19, 19, 19, 18, 19, 18, 19, 18,
                   19, 19, 19, 18, 19, 18, 19, 18,
                   19, 19, 19, 18, 19, 18, 19, 18});
  ms_setup(params, &idx, 32, 
           (int[]){64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                   64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                   64, 64});
  ms_setup(params, &idx, 64, 
           (int[]){3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 
                   3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 
                   3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                   3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2});
  ms_setup(params, &idx, 64, 
           (int[]){10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 10, 9, 10, 9, 9, 9,
                   10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9,
                   10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9,
                   10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 9});
  ms_setup(params, &idx, 64, 
           (int[]){32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32});

  int new = generate_block_size_params_uneven(params + actual);

  check_n_params(actual + new, n_params); check_n_params(idx + new, n_params);

  return cr_make_param_array(struct ParametersBlockSizes, params, n_params, 
                             free_block_params);
}


ParameterizedTest(struct ParametersBlockSizes *params, constructors,
                  block_sizes_custom) {
  cr_log_info("height=%d, m=%d", params->height, params->expected[0]);

  int ierr = SUCCESS;
  
  struct TreeHODLR *hodlr = 
    allocate_tree_monolithic(params->height, &ierr, &malloc, &free);
  if (hodlr == NULL) {
    cr_fail("Allocation failed");
    return;
  }

  struct HODLRInternalNode **queue = 
    compute_block_sizes_custom(hodlr, params->ms);

  for (int parent = 0; parent < hodlr->len_work_queue; parent++) {
    cr_expect(eq(ptr, queue[parent], 
                 hodlr->innermost_leaves[2 * parent]->parent));
  }

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
      expect_block_sizes(params->expected[eidx], params->expected[eidx+1], 
                         queue, height, qidx, eidx+2, true);
      eidx += 2;

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[qidx]->children[3].internal;
      queue[qidx] = queue[qidx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  for (int parent = 0; parent < len_queue; parent++) {
    const int m_larger = params->expected[eidx]; eidx++;
    const int m_smaller = params->expected[eidx]; eidx++;
    expect_block_sizes(m_larger, m_smaller, queue, -1, parent, eidx, false);
  }

  free_tree_hodlr(&hodlr, &free);
}


struct ParametersCopyDiag {
  int m;
  double *matrix;
  long len;
  double **expected_data;
  int *block_ms;
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
    cr_free(param->block_ms);
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


static inline double * laplacian_wrapper(int m, double _) {
  return construct_laplacian_matrix(m);
}


static inline int generate_copy_diagonal_params_halves(
  struct ParametersCopyDiag *params,
  double *(*func)(int, double)
) {
  const int len_ms = 3;
  const int ms[] = {21, 33, 64};

  const int len_lens = 3;
  const int lens[] = {2, 4, 8};

  int idx = 0;

  for (int midx = 0; midx < len_ms; midx++) {
    for (int lidx = 0; lidx < len_lens; lidx++) {
      const int m = ms[midx];
      const int len = lens[lidx];

      params[idx].m = m;
      params[idx].len = len;
      params[idx].matrix = func(m, 0.0);

      const int m_smaller = m / len;
      params[idx].block_ms = cr_malloc(len * sizeof(int));

      const int first_two = 2 * m_smaller + (m - len * m_smaller);
      params[idx].block_ms[1] = first_two / 2;
      params[idx].block_ms[0] = first_two - params[idx].block_ms[1];
      for (int i = 2; i < len; i++) {
        params[idx].block_ms[i] = m_smaller;
      }

      double start = 0.0;
      params[idx].expected_data = cr_malloc(len * sizeof(double *));
      for (int i = 0; i < len; i++) {
        params[idx].expected_data[i] = func(params[idx].block_ms[i], start);
        start += params[idx].block_ms[i];
      }
      idx++;
    }
  }

  check_n_params(idx, len_ms * len_lens);
  return len_ms * len_lens;
}


CREATE_SETUP_FUNC(ParametersCopyDiag, block_ms)


static inline int generate_copy_diagonal_params_custom(
  struct ParametersCopyDiag *params
) {
  enum {n_params = 4};

  const int ms[n_params] = {10, 22, 42, 128};
  const long lens[n_params] = {2, 4, 8, 16};

  int idx = 0;
  block_ms_setup(params, &idx, 2, (int[]){6, 4});
  block_ms_setup(params, &idx, 4, (int[]){20, 1, 1, 0});
  block_ms_setup(params, &idx, 8, (int[]){3, 21, 6, 1, 4, 3, 3, 1});
  block_ms_setup(params, &idx, 16, 
                   (int[]){4, 16, 8, 4, 8, 4, 4, 8, 16, 8, 8, 8, 16, 4, 8, 4});

  check_n_params(idx, n_params);

  for (int i = 0; i < n_params; i++) {
    params[i].m = ms[i];
    params[i].len = lens[i];
    params[i].matrix = construct_laplacian_matrix(ms[i]);

    params[i].expected_data = cr_malloc(lens[i] * sizeof(double *));
    for (int block = 0; block < lens[i]; block++) {
      params[i].expected_data[block] = 
        construct_laplacian_matrix(params[i].block_ms[block]);
    }
  }

  return n_params;
}


ParameterizedTestParameters(constructors, copy_diagonal) {
  const int n_params = 2*9 + 4;
  struct ParametersCopyDiag *params = 
    cr_malloc(n_params * sizeof(struct ParametersCopyDiag));

  int actual_n_params = 
    generate_copy_diagonal_params_halves(params, &laplacian_wrapper);

  actual_n_params += generate_copy_diagonal_params_halves(
    params + actual_n_params, &construct_diagonal_increasing
  );

  actual_n_params += 
    generate_copy_diagonal_params_custom(params + actual_n_params);

  check_n_params(actual_n_params, n_params);

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
    queue[i]->children[0].leaf->data.diagonal.m = params->block_ms[2 * i];

    queue[i]->children[3].leaf = malloc(sizeof(struct HODLRLeafNode));
    queue[i]->children[3].leaf->data.diagonal.m = params->block_ms[2 * i + 1];
  }

  int ierr = SUCCESS, idx = 0;
  copy_diagonal_blocks(params->matrix, params->m, queue, n_parent_nodes, &ierr,
                       &malloc);

  cr_expect(eq(int, ierr, SUCCESS));
  
  for (int i = 0; i < n_parent_nodes; i++) {
    for (int child = 0; child < 4; child += 3) {
      if (queue[i]->children[child].leaf->data.diagonal.data == NULL) {
        cr_fail("Data matrix for node %d (parent=%d, leaf=0) is NULL", idx, i);
        free(queue[i]->children[child].leaf);
        continue;
      }
      
      expect_matrix_double_eq(
        queue[i]->children[child].leaf->data.diagonal.data,
        params->expected_data[idx],
        queue[i]->children[child].leaf->data.diagonal.m,
        queue[i]->children[child].leaf->data.diagonal.m,
        queue[i]->children[child].leaf->data.diagonal.m,
        params->block_ms[idx],
        idx, NULL, NULL
      );
      idx++;

      free(queue[i]->children[child].leaf->data.diagonal.data);
      free(queue[i]->children[child].leaf);
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
  char name[STR_LEN];
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
  enum {n_params = 7};
  *len = n_params;
  struct ParametersTestCompress *params = 
    cr_malloc(n_params * sizeof(struct ParametersTestCompress));

  const double svd_threshold = 1e-8;
  const int m_fulls[n_params] = {10, 11, 11, 10, 18, 7, 13};
  const int ms[n_params] = {5, 5, 6, 5, 5, 7, 4};
  const int ns[n_params] = {5, 6, 5, 5, 4, 7, 9};
  const int ss[n_params] = {1, 1, 2, 2, 1, 1, 1};

  for (int i = 0; i < n_params; i++) {
    params[i].m_full = m_fulls[i];
    params[i].m = ms[i];
    params[i].n = ns[i];
    params[i].svd_threshold = svd_threshold;
    params[i].expected_n_singular = ss[i];
    params[i].u_expected = 
      cr_calloc(params[i].m * params[i].expected_n_singular, sizeof(double));
    params[i].v_expected = 
      cr_calloc(params[i].expected_n_singular * params[i].n, sizeof(double));
  }

  int idx = 0;

  // Laplacian matrix bottom left quarter
  strncat(params[idx].name, "L_bl", STR_LEN);
  params[idx].full_matrix = construct_laplacian_matrix(params[idx].m_full);
  params[idx].matrix = params[idx].full_matrix + params[idx].m;
  params[idx].u_expected[0] = 1.0;
  params[idx].v_expected[0] = -0.0;
  params[idx].v_expected[params[idx].n - 1] = -1.0;
  idx++;

  // Laplacian matrix top right quarter
  strncat(params[idx].name, "L_tr", STR_LEN);
  params[idx].full_matrix = construct_laplacian_matrix(params[idx].m_full);
  params[idx].matrix = 
    params[idx].full_matrix + params[idx].m_full * params[idx].m;
  params[idx].u_expected[params[idx].m - 1] = -1.0;
  params[idx].v_expected[0] = 1.0;
  idx++;

  for (int i = idx; i < idx+2; i++) {
    params[i].full_matrix = construct_laplacian_matrix(params[i].m_full);
    params[i].full_matrix[params[i].m_full - 1] = 0.5;
    params[i].full_matrix[params[i].m_full * (params[i].m_full - 1)] = 0.5;
  }

  // Laplacian matrix with corners bottom left quarter
  strncat(params[idx].name, "L0.5S_bl", STR_LEN);
  params[idx].matrix = params[idx].full_matrix + params[idx].n;
  params[idx].u_expected[0] = 1.0;
  params[idx].u_expected[2 * params[idx].m - 1] = -0.5;

  params[idx].v_expected[params[idx].n - 1] = -1.0;
  params[idx].v_expected[params[idx].n] = -1.0;
  idx++;

  // Laplacian matrix with corners top right quarter
  strncat(params[idx].name, "L0.5S_tr", STR_LEN);
  params[idx].matrix = 
    params[idx].full_matrix + params[idx].m_full * params[idx].m;
  params[idx].u_expected[params[idx].m - 1] = 1.0;
  params[idx].u_expected[params[idx].m] = 0.5;

  params[idx].v_expected[0] = -1.0;
  params[idx].v_expected[2 * params[idx].n - 1] = 1.0;
  idx++;

  // Identity matrix
  strncat(params[idx].name, "I", STR_LEN);
  params[idx].full_matrix = construct_identity_matrix(params[idx].m_full);
  params[idx].matrix = params[idx].full_matrix + params[idx].m;
  params[idx].v_expected[0] = 1.0;
  idx++;

  // Zeros matrix
  strncat(params[idx].name, "0", STR_LEN);
  params[idx].full_matrix = cr_calloc(params[idx].m_full * params[idx].m_full, 
                                      sizeof(double));
  params[idx].matrix = params[idx].full_matrix;
  params[idx].v_expected[0] = 1.0;
  idx++;

  // Laplacian matrix bottom left quarter uneven
  strncat(params[idx].name, "L_skew", STR_LEN);
  params[idx].full_matrix = construct_laplacian_matrix(params[idx].m_full);
  params[idx].matrix = params[idx].full_matrix + params[idx].n;
  params[idx].u_expected[0] = 1.0;
  params[idx].v_expected[params[idx].n - 1] = -1.0;
  idx++;

  check_n_params(idx, n_params);

  return params;
}


ParameterizedTestParameters(constructors, test_compress) {
  int n_params;
  struct ParametersTestCompress *params = generate_compress_params(&n_params);
  return cr_make_param_array(struct ParametersTestCompress, params, n_params, 
                             free_compress_params);
}


ParameterizedTest(struct ParametersTestCompress *params, constructors, 
                  test_compress) {
  cr_log_info("%s: m=%d, n=%d, expected s=%d", 
              params->name, params->m, params->n, params->expected_n_singular);

  struct NodeOffDiagonal result;
  result.m = params->m; result.n = params->n;

  int ierr = SUCCESS;
  const int n_singular_values = params->m < params->n ? params->m : params->n;
  
  double *s_work = malloc(n_singular_values * sizeof(double));
  double *u_work = malloc(params->m * n_singular_values * sizeof(double));
  double *vt_work = malloc(params->n * n_singular_values * sizeof(double));

  int result_code = compress_off_diagonal(
      &result, n_singular_values, params->m_full, 
      params->matrix, s_work, u_work, vt_work, params->svd_threshold, &ierr,
      &malloc
  );

  free(s_work); free(u_work); free(vt_work);

  cr_expect(eq(int, ierr, SUCCESS));
  cr_expect(eq(int, result_code, 0));
  if (result_code != 0 || ierr != SUCCESS) {
    free(result.u); free(result.v);
    cr_fatal();
  } 
  cr_expect(eq(int, result.s, params->expected_n_singular));

  expect_matrix_double_eq(
    result.u, params->u_expected, params->m, params->expected_n_singular,
    result.m, params->m, 'U', NULL, NULL
  );
  expect_matrix_double_eq(
    result.v, params->v_expected, params->n, params->expected_n_singular,
    result.m, params->m, 'V', NULL, NULL
  );
  free(result.u); free(result.v);
}


ParameterizedTestParameters(constructors, recompress) {
  int n_params;
  struct ParametersTestCompress *params = generate_compress_params(&n_params);
  return cr_make_param_array(struct ParametersTestCompress, params, n_params, 
                             free_compress_params);
}


ParameterizedTest(struct ParametersTestCompress *params, 
                  constructors, recompress) {
  cr_log_info("%s: m=%d, n=%d, expected s=%d", 
              params->name, params->m, params->n, params->expected_n_singular);

  struct NodeOffDiagonal node;
  node.m = params->m; node.n = params->n;

  int ierr = SUCCESS;
  const double alpha = 1, beta = 0;

  const int n_singular_values = params->m < params->n ? params->m : params->n;
  
  int diff = params->matrix - params->full_matrix;
  double *og_data = malloc(params->m_full * params->m_full * sizeof(double));
  memcpy(og_data, params->full_matrix, params->m_full * params->m_full * sizeof(double));
  double *og_matrix = og_data + diff;

  double *s_work = malloc(n_singular_values * sizeof(double));
  double *u_work = malloc(params->m * n_singular_values * sizeof(double));
  double *vt_work = malloc(params->n * n_singular_values * sizeof(double));

  int result_code = compress_off_diagonal(
      &node, n_singular_values, params->m_full, 
      params->matrix, s_work, u_work, vt_work, params->svd_threshold,
      &ierr, &malloc
  );
  
  free(s_work); free(u_work); free(vt_work);
  
  cr_expect(eq(int, ierr, SUCCESS));
  cr_expect(eq(int, result_code, 0));
  if (result_code != 0 || ierr != SUCCESS) {
    free(og_data);
    cr_fatal();
  } 

  double *result = malloc(params->m * params->n * sizeof(double));
  dgemm_("N", "T", &node.m, &node.n, 
         &node.s, &alpha, node.u, &node.m, 
         node.v, &node.n, &beta, result, &params->m);
  free(node.u); free(node.v);

  double norm, diffd;
  expect_matrix_double_eq(result, og_matrix, node.m, node.n, 
                          node.m, params->m_full, 'A', &norm, &diffd);
  cr_log_info("normv=%f, diff=%f, relerr=%f", sqrtf(norm), sqrtf(diffd),
              sqrtf(diffd) / sqrtf(norm));

  free(og_data); free(result);
}


struct ParametersTestDense {
  double *matrix;
  int m;
  int *ms;
  int height;
  double svd_threshold;
  struct TreeHODLR *expected;
  char name[STR_LEN];
};


void free_dense_params(struct criterion_test_params *params) {
  for (size_t i = 0; i < params->length; i++) {
    struct ParametersTestDense *param = 
      (struct ParametersTestDense *)params->params + i;
    cr_free(param->matrix);
    cr_free(param->ms);

    cr_free(param->expected->innermost_leaves[0]->data.diagonal.data);
    cr_free(param->expected->innermost_leaves);
    cr_free(param->expected->work_queue);
    cr_free(param->expected->memory_internal_ptr);
    cr_free(param->expected->memory_leaf_ptr);
    cr_free(param->expected);
  }
  cr_free(params->params);
}


static inline int * arrcpy(const int len, const int src[]) {
  int *dest = cr_malloc(len * sizeof(int));
  for (int i = 0; i < len; i++) {
    dest[i] = src[i];
  }
  return dest;
}


static inline void allocate_dense_params(
  struct ParametersTestDense *params,
  const int max_height,
  const int n_types,
  const int m,
  const int *ms[]
) {
  unsigned int idx = 0; int ierr = SUCCESS;

  for (int mult = 0; mult < n_types; mult++) {
    for (int height = 1; height < max_height + 1; height++) {
      params[idx].height = height;
      params[idx].m = m;
      params[idx].ms = (int*)ms[idx];
      params[idx].svd_threshold = 1e-8;
      params[idx].expected = 
        allocate_tree_monolithic(height, &ierr, &cr_malloc, &cr_free);

      if (ms[idx] == NULL) {
        compute_block_sizes_halves(params[idx].expected, m);
      } else {
        compute_block_sizes_custom(params[idx].expected, ms[idx]);
      }
      idx++;
    }
  }

  check_n_params(idx, max_height * n_types);
}


static inline int generate_dense_params(struct ParametersTestDense *params) {
  enum {n_params = 10};
  
  int idx = 0;
  const int m = 69;
  const int *ms[n_params] = {
    NULL, NULL, NULL, NULL, NULL,
    arrcpy(2, (int[]){29, 40}),
    arrcpy(4, (int[]){21, 10, 5, 33}),
    arrcpy(8, (int[]){10, 11, 1, 9, 5, 20, 7, 6}),
    arrcpy(16, (int[]){2, 8, 7, 4, 1, 5, 4, 2, 3, 10, 5, 6, 5, 2, 3, 2}),
    arrcpy(32, (int[]){1, 1, 5, 3, 2, 5, 2, 2, 1, 1, 4, 2, 2, 1, 1, 2, 
                       1, 4, 3, 3, 1, 3, 5, 1, 2, 3, 1, 1, 2, 1, 1, 2}),
  };

  const int n_types = 2, max_height = 5;
  allocate_dense_params(params, max_height, n_types, m, ms);

  for (int mult = 0; mult < n_types; mult++) {
    for (int height = 1; height < max_height + 1; height++) {
      strncat(params[idx].name, "L0.5S", STR_LEN);
      params[idx].matrix = construct_laplacian_matrix(m);
      params[idx].matrix[m - 1] = 0.5;
      params[idx].matrix[m * (m - 1)] = 0.5;

      double *matrix = construct_laplacian_matrix(m);
      matrix[m - 1] = 0.5;
      matrix[m * (m - 1)] = 0.5;
      construct_fake_hodlr(params[idx].expected, matrix, 1, NULL);

      params[idx].expected->root->children[1].leaf->data.off_diagonal.s = 2;
      params[idx].expected->root->children[2].leaf->data.off_diagonal.s = 2;
      idx++;
    }
  }

  check_n_params(idx, n_params);
  return n_params;
}


static inline int generate_dense_params_decay(
  struct ParametersTestDense *const params
) {
  enum {n_params = 20};
  
  int idx = 0;
  const int m = 155;
  const int *ms[n_params] = {
    NULL, NULL, NULL, NULL, NULL,
    arrcpy(2, (int[]){55, 100}),
    arrcpy(4, (int[]){50, 5, 50, 50}),
    arrcpy(8, (int[]){22, 11, 11, 11, 25, 25, 25, 25}),
    arrcpy(16, (int[]){13, 15, 10, 4, 15, 12, 6, 3, 
                       6, 14, 2, 4, 23, 11, 11, 6}),
    arrcpy(32, (int[]){2, 6, 8, 2, 2, 6, 4, 6, 8, 7, 6, 7, 4, 5, 6, 7, 
                       8, 4, 2, 3, 7, 4, 5, 2, 4, 4, 3, 6, 2, 9, 2, 4}),
    NULL, NULL, NULL, NULL, NULL,
    arrcpy(2, (int[]){55, 100}),
    arrcpy(4, (int[]){50, 5, 50, 50}),
    arrcpy(8, (int[]){22, 11, 11, 11, 25, 25, 25, 25}),
    arrcpy(16, (int[]){13, 15, 10, 4, 15, 12, 6, 3, 
                       6, 14, 2, 4, 23, 11, 11, 6}),
    arrcpy(32, (int[]){2, 6, 8, 2, 1, 6, 4, 6, 8, 7, 6, 7, 4, 6, 6, 7, 
                       8, 4, 2, 3, 7, 4, 5, 2, 4, 4, 3, 6, 2, 9, 1, 5}),
  };

  const int n_types = 2;
  const int n_repeats = 2 * n_types, max_height = 5;
  allocate_dense_params(params, max_height, n_repeats, m, ms);

  srand(42);
  for (int mult = 0; mult < n_types; mult++) {
    for (int height = 1; height < max_height + 1; height++) {
      strncat(params[idx].name, "DecaySort", STR_LEN);
      params[idx].matrix = cr_malloc(m * m * sizeof(double));
      fill_decay_matrix_random_sorted(m, 1.0, params[idx].matrix);

      double *matrix = cr_malloc(m * m * sizeof(double));
      memcpy(matrix, params[idx].matrix, m * m * sizeof(double));
      construct_fake_hodlr(params[idx].expected, matrix, 1, NULL);
      idx++;
    }
  }
  for (int mult = 0; mult < n_types; mult++) {
    for (int height = 1; height < max_height + 1; height++) {
      strncat(params[idx].name, "Decay", STR_LEN);
      params[idx].matrix = cr_malloc(m * m * sizeof(double));
      fill_decay_matrix_random(m, 1.0, params[idx].matrix);

      double *matrix = cr_malloc(m * m * sizeof(double));
      memcpy(matrix, params[idx].matrix, m * m * sizeof(double));
      construct_fake_hodlr(params[idx].expected, matrix, 0, NULL);
      idx++;
    }
  }

  check_n_params(idx, n_params);
  return n_params;
}


ParameterizedTestParameters(constructors, dense_to_tree) {
  const int n_params = 10+2*10;
  struct ParametersTestDense *params = 
    cr_malloc(n_params * sizeof(struct ParametersTestDense));

  int actual = generate_dense_params(params);
  actual += generate_dense_params_decay(params + actual);

  check_n_params(actual, n_params);
  return cr_make_param_array(struct ParametersTestDense, params, n_params, 
                             free_dense_params);
}


ParameterizedTest(struct ParametersTestDense *params, 
                  constructors, dense_to_tree) {
  int ierr;
  cr_log_info("matrix %s height=%d, m=%d, ms=%p", 
              params->name, params->height, params->m, params->ms);

  struct TreeHODLR *result = allocate_tree_monolithic(params->height, &ierr,
                                                      &malloc, &free);
  cr_expect(eq(int, ierr, SUCCESS));
  cr_expect(ne(ptr, result, NULL));
  if (ierr != SUCCESS) {
    cr_fatal("Tree HODLR allocation failed");
  }

  int svd = dense_to_tree_hodlr(
    result, params->m, params->ms, params->matrix, params->svd_threshold, 
    &ierr, &malloc, &free
  );
  
  cr_expect(eq(int, ierr, SUCCESS));
  cr_expect(zero(svd));
  if (ierr != SUCCESS) {
    free_tree_hodlr(&result, &free);
    cr_fatal();
  }

  const int m = result->root->children[1].leaf->data.off_diagonal.m;
  const int n = result->root->children[1].leaf->data.off_diagonal.n;
  double *workspace = malloc(m * n * sizeof(double));

  expect_hodlr_decompress(
    true, result, params->expected, workspace, NULL, NULL, NULL, DELTA
  );

  free_tree_hodlr(&result, &free); free(workspace);
}

