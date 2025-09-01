#include <math.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "../include/common_data.h"
#include "../../include/hmat_lib/hodlr.h"


double * construct_any_matrix(const int m, 
                              void(*matrix_func)(const int, double *)) {
  double *matrix = cr_malloc(m * m * sizeof(double));
  matrix_func(m, matrix);
  return matrix;
}


double * construct_laplacian_matrix(int m) {
  double *matrix = cr_malloc(m * m * sizeof(double));

  fill_laplacian_matrix(m, matrix);

  return matrix;
}


void fill_laplacian_matrix(const int m, double *matrix) {
  int idx;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      idx = j + i * m;
      if (i == j) {
        matrix[idx] = 2.0;
      } else if (i == j+1 || i == j-1) {
        matrix[idx] = -1.0;
      } else {
        matrix[idx] = 0.0;
      }
    }
  }
}


void fill_laplacian_converse_matrix(const int m, double *matrix) {
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      if (i == j) {
        matrix[i + j * m] = -1.0;
      } else if (i == j + 1 || i == j - 1) {
        matrix[i + j * m] = 2.0;
      } else {
        matrix[i + j * m] = 0.0;
      }
    }
  }
}


double * construct_identity_matrix(int m) {
  double *matrix = cr_malloc(m * m * sizeof(double));
  fill_identity_matrix(m, matrix);
  return matrix;
}


void fill_identity_matrix(const int m, double *matrix) {
  int idx;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      idx = j + i * m;
      if (i == j) {
        matrix[idx] = 1;
      } else {
        matrix[idx] = 0;
      }
    }
  }
}


double * construct_full_matrix(const int m, const double val) {
  double *matrix = cr_malloc(m * m * sizeof(double));
  fill_full_matrix(m, val, matrix);
  return matrix;
}


void fill_full_matrix(const int m, const double val, double *matrix) {
  int idx = 0, i = 0, j = 0;
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j ++) {
      matrix[i + j * m] = val;
    }
  }
}


double * construct_random_matrix(const int m, const int n) {
  double * matrix = cr_malloc(m * n * sizeof(double));
  fill_random_matrix(m, n, matrix);
  return matrix;
}


void fill_random_matrix(const int m, const int n, double *matrix) {
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      matrix[i + j * m] = (double)rand() / RAND_MAX;
    }
  }
}


void fill_decay_matrix(
  const int m, 
  const double *restrict const vec, 
  const double scaling_factor,
  double *restrict const matrix
) {
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      double val = vec[i] - vec[j];
      matrix[i + j * m] = exp(- scaling_factor * fabs(val));
    }
  }
}


void fill_decay_matrix_random(
  const int m,
  const double scaling_factor,
  double *restrict const matrix
) {
  double *vec = cr_malloc(m * sizeof(double));
  for (int i = 0; i < m; i++)
    vec[i] = 1.0 - 2.0 * ((double)rand() / RAND_MAX);
  fill_decay_matrix(m, vec, scaling_factor, matrix);
  cr_free(vec);
}


static int compare_double(const void* p1, const void* p2) {
  if (*(double*)p1 < *(double*)p2) return -1;
  if (*(double*)p1 > *(double*)p2) return 1;
  return 0;
}


void fill_decay_matrix_random_sorted(
  const int m,
  const double scaling_factor,
  double *restrict const matrix
) {
  double *vec = cr_malloc(m * sizeof(double));
  for (int i = 0; i < m; i++)
    vec[i] = 1.0 - 2.0 * ((double)rand() / RAND_MAX);
  qsort(vec, m, sizeof(double), compare_double);
  fill_decay_matrix(m, vec, scaling_factor, matrix);
  cr_free(vec);
}


void construct_fake_hodlr(struct TreeHODLR *restrict const hodlr, 
                          double *restrict const matrix,
                          const int s,
                          const int *restrict const ss) {
  struct HODLRInternalNode **queue = hodlr->work_queue;
  long n_parent_nodes = hodlr->len_work_queue;
  const int matrix_ld = hodlr->root->m;

  int offset = 0, idx = 0;
  for (int parent = 0; parent < n_parent_nodes; parent++) {
    queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;
    
    const int m = hodlr->innermost_leaves[2 * parent]->data.diagonal.m;
    const int n = hodlr->innermost_leaves[2 * parent + 1]->data.diagonal.m;

    hodlr->innermost_leaves[2 * parent]->data.diagonal.data = 
      matrix + offset + offset * matrix_ld;

    queue[parent]->children[1].leaf->data.off_diagonal.u =
      matrix + offset + (offset + m) * matrix_ld;

    queue[parent]->children[2].leaf->data.off_diagonal.u =
      matrix + offset + m + offset * matrix_ld;

    if (ss == NULL) {
      queue[parent]->children[1].leaf->data.off_diagonal.s = s;
      queue[parent]->children[2].leaf->data.off_diagonal.s = s;
    } else {
      queue[parent]->children[1].leaf->data.off_diagonal.s = ss[idx]; idx++;
      queue[parent]->children[2].leaf->data.off_diagonal.s = ss[idx]; idx++;
    }

    offset += m;
    hodlr->innermost_leaves[2 * parent + 1]->data.diagonal.data = 
      matrix + offset + offset * matrix_ld;

    offset += n;
  }

  for (int level = hodlr->height; level > 1; level--) {
    n_parent_nodes /= 2;
    int offset = 0;

    for (int parent = 0; parent < n_parent_nodes; parent++) {
      const int m = queue[2 * parent]->m, n = queue[2 * parent + 1]->m;
      queue[parent] = queue[2 * parent]->parent;

      queue[parent]->children[1].leaf->data.off_diagonal.u =
        matrix + offset + (offset + m) * matrix_ld;

      queue[parent]->children[2].leaf->data.off_diagonal.u =
        matrix + offset + m + offset * matrix_ld;

      if (ss == NULL) {
        queue[parent]->children[1].leaf->data.off_diagonal.s = s;
        queue[parent]->children[2].leaf->data.off_diagonal.s = s;
      } else {
        queue[parent]->children[1].leaf->data.off_diagonal.s = ss[idx]; idx++;
        queue[parent]->children[2].leaf->data.off_diagonal.s = ss[idx]; idx++;
      }

      offset += m + n;
    }
  }
}


void fill_tridiag_symmetric1_matrix(const int m, double *matrix) {
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      if (i == j) matrix[i + j * m] = (double)(m - i);
      else if (i == j - 1 || i == j + 1) {
        matrix[i + j * m] = (double)(i * j);
      } else {
        matrix[i + j * m] = 0.0;
      }
    }
  }
}

