#include <math.h>
#include <stdio.h>
#include <string.h>

#include <criterion/criterion.h>
#include <criterion/logging.h>
#include <criterion/new/assert.h>

#include "../include/utils.h"
#include "../../include/tree.h"

static double DELTA = 1e-10;


inline void expect_leaf_offdiagonal(struct HODLRLeafNode *leaf,
                                    struct HODLRInternalNode *parent) {
  cr_expect(eq(ptr, leaf->parent, parent));
  cr_expect(eq(int, leaf->type, OFFDIAGONAL));
  cr_expect(eq(ptr, leaf->data.off_diagonal.u, NULL));
  cr_expect(eq(ptr, leaf->data.off_diagonal.v, NULL));
  cr_expect(eq(int, leaf->data.off_diagonal.m, 0));
  cr_expect(eq(int, leaf->data.off_diagonal.s, 0));
  cr_expect(eq(int, leaf->data.off_diagonal.n, 0));
}


inline void expect_leaf_diagonal(struct HODLRLeafNode *leaf,
                                 struct HODLRInternalNode *parent) {
  cr_expect(eq(ptr, leaf->parent, parent));
  cr_expect(eq(int, leaf->type, DIAGONAL));
  cr_expect(eq(ptr, leaf->data.diagonal.data, NULL));
  cr_expect(eq(int, leaf->data.diagonal.m, 0));
}


inline void expect_internal(struct HODLRInternalNode *node,
                            struct HODLRInternalNode *parent) {
  cr_expect(eq(ptr, node->parent, parent));
  cr_expect(eq(int, node->m, 0));
}


int expect_tree_consistent(struct TreeHODLR *hodlr, 
                           int height,
                           const long max_depth_n) {
  int len_queue = 1;

  cr_expect(eq(i32, hodlr->height, height));
  cr_expect(eq(hodlr->len_work_queue, max_depth_n));
  cr_expect(ne(ptr, hodlr->work_queue, NULL));

  cr_expect(eq(ptr, hodlr->root->parent, NULL));
  cr_expect(eq(int, hodlr->root->m, 0));

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **temp_pointer;
  queue[0] = hodlr->root;

  for (int i = 1; i < height; i++) {
    for (int j = 0; j < len_queue; j++) {
      expect_internal(queue[j]->children[0].internal, queue[j]);
      expect_leaf_offdiagonal(queue[j]->children[1].leaf, queue[j]);
      expect_leaf_offdiagonal(queue[j]->children[2].leaf, queue[j]);
      expect_internal(queue[j]->children[3].internal, queue[j]);

      next_level[2 * j] = queue[j]->children[0].internal;
      next_level[2 * j + 1] = queue[j]->children[3].internal;

    }
    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;

    len_queue = len_queue * 2;
  }

  for (int i = 0; i < len_queue; i++) {
    expect_leaf_diagonal(queue[i]->children[0].leaf, queue[i]);
    expect_leaf_offdiagonal(queue[i]->children[1].leaf, queue[i]);
    expect_leaf_offdiagonal(queue[i]->children[2].leaf, queue[i]);
    expect_leaf_diagonal(queue[i]->children[3].leaf, queue[i]);

    cr_expect(eq(ptr, hodlr->innermost_leaves[2 * i], queue[i]->children[0].leaf));
    cr_expect(eq(ptr, hodlr->innermost_leaves[2 * i + 1], queue[i]->children[3].leaf));
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

        err += expect_matrix_double_eq_safe(act->off_diagonal.u, exp->off_diagonal.u,
                                         act->off_diagonal.m, act->off_diagonal.s,
                                         exp->off_diagonal.m, exp->off_diagonal.s,
                                         act->off_diagonal.m, exp->off_diagonal.m,
                                         'U');
        
        err += expect_matrix_double_eq_safe(act->off_diagonal.v, exp->off_diagonal.v,
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
      err += expect_matrix_double_eq_safe(act->off_diagonal.u, exp->off_diagonal.u,
                                       act->off_diagonal.m, act->off_diagonal.s,
                                       exp->off_diagonal.m, exp->off_diagonal.s,
                                       act->off_diagonal.m, exp->off_diagonal.m,
                                       'U');
      err += expect_matrix_double_eq_safe(act->off_diagonal.v, exp->off_diagonal.v,
                                       act->off_diagonal.n, exp->off_diagonal.s,
                                       exp->off_diagonal.n, exp->off_diagonal.s,
                                       act->off_diagonal.n, exp->off_diagonal.n,
                                       'V');
    }
    act = &(queue_a[i]->children[0].leaf->data);
    exp = &(queue_e[i]->children[0].leaf->data);

    cr_expect(eq(act->diagonal.m, exp->diagonal.m));

    if (act->diagonal.m == exp->diagonal.m) {
      expect_matrix_double_eq(act->diagonal.data, exp->diagonal.data, 
                           exp->diagonal.m, exp->diagonal.m, 
                           act->diagonal.m, exp->diagonal.m, 'M');
    }

    act = &(queue_a[i]->children[3].leaf->data);
    exp = &(queue_e[i]->children[3].leaf->data);

    cr_expect(eq(act->diagonal.m, exp->diagonal.m));

    if (act->diagonal.m == exp->diagonal.m) {
      expect_matrix_double_eq(act->diagonal.data, exp->diagonal.data, 
                           exp->diagonal.m, exp->diagonal.m, 
                           act->diagonal.m, exp->diagonal.m, 'M');
    } 
  }

  free(queue_a); free(next_level_a); free(queue_e); free(next_level_e);
  return err;
}




void log_matrix(const double *matrix, const int m, const int n, const int lda) {
  for (int i = 0; i < m; i++) {
    char *buffer = malloc(n * 16 * sizeof(char));
    buffer[0] = '\0';

    for (int j = 0; j < n; j++) {
      char temp[16];
      snprintf(temp, sizeof(temp), "%12.5e  ", matrix[j * lda + i]);
      strcat(buffer, temp);
      //printf("%f    ", matrix[j * lda + i]);
    }
    //printf("\n");
    cr_log_info("%s", buffer);
    free(buffer);
  }
  //printf("\n");
}


int expect_matrix_double_eq_safe(
  const double *restrict actual, 
  const double *restrict expected, 
  const int m_actual, 
  const int n_actual,
  const int m_expected, 
  const int n_expected,
  const int ld_actual, 
  const int ld_expected,
  const char name)
{
  int err = 0;
  if (m_actual != m_expected) {
    err = 1;
    cr_fail("actual matrix dimension 1 (M) different than expected (actual=%d vs expected=%d)",
            m_actual, m_expected);
  }
  
  if (n_actual != n_expected) {
    err = 1;
    cr_fail("actual matrix dimension 2 (N) different than expected (actual=%d vs expected=%d)",
            n_actual, n_expected);
  }
  if (err == 0) {
    expect_matrix_double_eq(actual, expected, m_expected, n_expected, ld_actual, ld_expected, name);
  }

  return err;
}


void expect_matrix_double_eq(const double *restrict actual, const double *restrict expected, 
                          const int m, const int n,
                          const int ld_actual, const int ld_expected,
                          const char name) {

  int errors = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (fabs(actual[j + i * ld_actual] - expected[j + i * ld_expected]) >
          DELTA) {
        cr_log_error("actual value '%f' at index [%d, %d] is different from "
                     "the expected '%f'",
                     actual[j + i * ld_actual], j, i,
                     expected[j + i * ld_expected]);
        errors += 1;
      }
    }
  }

  if (errors > 0) {
    cr_fail("The %c matrices are not equal (%d errors)", name, errors);
    cr_log_info("Actual:");
    log_matrix(actual, m, n, ld_actual);
    cr_log_info("Expected:");
    log_matrix(expected, m, n, ld_expected);
  }
}


void log_vector(const double *restrict vector, const int len) {
  char *buffer = malloc(len * 16 * sizeof(char));
  buffer[0] = '\0';

  for (int j = 0; j < len; j++) {
    char temp[16];
    snprintf(temp, sizeof(temp), "%12.5e  ", vector[j]);
    strcat(buffer, temp);
    //printf("%f    ", matrix[j * lda + i]);
  }
  //printf("\n");
  cr_log_info("%s", buffer);
  free(buffer);
}


int expect_vector_double_eq_safe(
  const double *restrict actual,
  const double *restrict expected,
  const int len_actual,
  const int len_expected,
  const char name
) {
  int errors = 0;

  if (len_actual != len_expected) {
    cr_fail("actual vector (%c) length (%f) is different from expected (%f)",
            name, len_actual, len_expected);
    return 1;
  }

  for (int i = 0; i < len_expected; i++) {
    if (fabs(actual[i] - expected[i]) > DELTA) {
      cr_log_error("actual value '%f' at index [%d] is different from "
                   "the expected '%f'", actual[i], i, expected[i]);
      errors += 1;
    }
  }

  if (errors > 0) {
    cr_fail("The %c vectors are not equal (%d errors)", name, errors);
    cr_log_info("Actual:");
    log_vector(actual, len_actual);
    cr_log_info("Expected:");
    log_vector(expected, len_expected);
  }

  return errors;
}

