#include <stdio.h>
#include <stdlib.h>
#include "lapack_wrapper.h"
#include "tree.h"
// #include <clapack.h>
//


void print_matrix(int m, int n, double *matrix) {
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      printf("%f    ", matrix[j * m + i]);
    }
    printf("\n");
  }
  printf("\n");
}



int compress_off_diagonal(struct HODLRNode *node,
                          int m, 
                          int n, 
                          int n_singular_values,
                          double *lapack_matrix,
                          double *s,
                          double *u,
                          double *vt,
                          double svd_threshold) {
  int result = svd_double(m, n, n_singular_values, lapack_matrix, s, u, vt);
  //printf("svd result %d\n", result);
  if (result != 0) {
    return result;
  }

  double primary_s_fraction = 1 / s[0];
  int svd_cutoff_idx;
  for (svd_cutoff_idx=1; svd_cutoff_idx < n_singular_values; svd_cutoff_idx++) {
    if (s[svd_cutoff_idx] * primary_s_fraction < svd_threshold) {
      break;
    }
  }
  //printf("svd cut-off=%d, m=%d\n", svd_cutoff_idx, m);

  double *u_top_right = malloc(m * svd_cutoff_idx * sizeof(double));
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<m; j++) {
      //printf("i=%d, j=%d, idx=%d\n", i, j, j + i * m);
      u_top_right[j + i * m] = u[j + i * m] * s[i];
    }
  }
  //print_matrix(svd_cutoff_idx, m, u_top_right);


  double *v_top_right = malloc(svd_cutoff_idx * n * sizeof(double));
  for (int i=0; i<n; i++) {
    for (int j=0; j<svd_cutoff_idx; j++) {
      v_top_right[i + j * n] = vt[j + i * svd_cutoff_idx];
    }
  }
  //print_matrix(n, svd_cutoff_idx, v_top_right);

  node->data.off_diagonal.u = u_top_right;
  node->data.off_diagonal.v = v_top_right;

  node->data.off_diagonal.m = m;
  node->data.off_diagonal.s = n_singular_values;
  node->data.off_diagonal.n = n;

  return 0;
}



struct HODLRNode create_one_level(int m, int n, double *matrix, double svd_threshold) {
  struct HODLRNode node_top_left;
  struct HODLRNode node_top_right;
  struct HODLRNode node_bottom_left;
  struct HODLRNode node_bottom_right;

  int m_top = m / 2;
  int m_bottom = m - m_top;

  int n_left = n / 2;
  int n_right = n - n_left;

  if (m_top != n_left) {
    // error out
  }

  if (m_bottom != n_right) {
    // error out
  }

  //printf("m_top=%d, m_bottom=%d, n_left=%d, n_right=%d\n", m_top, m_bottom, n_left, n_right);

  // TOP LEFT QUARTER
  double *top_left_data = malloc(m_top * n_left * sizeof(double));
  for  (int j = 0; j < n_left; j++){
    for (int i = 0; i < m_top; i++) {
      top_left_data[i + j * m_top] = matrix[i + j * m];
    }
  }
  //node_top_left.child = NULL;
  node_top_left.data.diagonal.data = top_left_data;
  node_top_left.data.diagonal.m = m_top;

  //print_matrix(m_top, n_left, node_top_left.data.diagonal.data);

  // TOP RIGHT QUARTER
  double *lapack_matrix = malloc(m_top * n_right * sizeof(double));
  for (int i = 0; i < n_right; i++) {
    for (int j = 0; j < m_top; j++) {
      lapack_matrix[j + i * m_top] = matrix[j + (i+n_left) * m];
    }
  }
  int n_singular_values = m_top < n_right ? m_top : n_right;
  double *s = malloc(n_singular_values * sizeof(double));
  double *u = malloc(m_top * n_singular_values * sizeof(double));
  double *vt = malloc(n_singular_values * n_right * sizeof(double));

  //print_matrix(m_top, n_right, lapack_matrix);

  int result = compress_off_diagonal(&node_top_right, m_top, n_right, n_singular_values, lapack_matrix, s, u, vt, svd_threshold);
  if (result != 0) {
    // error out
    printf("compress 1 failed\n");
  }
  //node_top_right.child = NULL;
  //printf("compress 1 succeeded\n");

  // BOTTOM LEFT QUARTER
  n_singular_values = m_bottom < n_right ? m_bottom : n_right;
  for (int i=0; i<n_right; i++) {
    for (int j=0; j<m_bottom; j++) {
      lapack_matrix[j + i * m_bottom] = matrix[j + m_top + i * m];
    }
  }
  //print_matrix(m_bottom, n_left, lapack_matrix);


  result = compress_off_diagonal(&node_bottom_left, m_bottom, n_left, n_singular_values, lapack_matrix, s, u, vt, svd_threshold);
  if (result != 0) {
    // Error out
    printf("compress 2 failed\n");
  }
  //node_bottom_left.child = NULL;
  //printf("compress 2 succeeded\n");

  free(lapack_matrix); free(s); free(u); free(vt);

  // BOTTOM RIGHT QUARTER
  double *bottom_right_data = malloc(m_bottom * n_right * sizeof(double));
  for (int i = 0; i < n_right; i++) {
    for (int j = 0; j < m_bottom; j++) {
      bottom_right_data[j + i * m_bottom] = matrix[j + m_top + (i+n_left) * m];
    }
  }  
  //node_bottom_right.children = NULL;
  node_bottom_right.data.diagonal.data = bottom_right_data;
  node_bottom_right.data.diagonal.m = m_bottom;

  //print_matrix(m_bottom, n_right, node_bottom_right.data.diagonal.data);

  //node_top_left.next_sibling = &node_top_right;
  //node_top_right.next_sibling = &node_bottom_left;
  //node_bottom_left.next_sibling = &node_bottom_right;
  //node_bottom_right.next_sibling = NULL;

  node_top_left.type = DIAGONAL;
  node_top_right.type = OFFDIAGONAL;
  node_bottom_left.type = OFFDIAGONAL;
  node_bottom_right.type = DIAGONAL;

  //printf("all nodes created\n");
  return node_top_left;
}


struct TreeHODLR dense_to_tree_hodlr(int m, 
                                     int n,
                                     double *matrix,
                                     double svd_threshold,
                                     int depth) {
  struct TreeHODLR root;

  struct HODLRNode child = create_one_level(m, n, matrix, svd_threshold);
  root.child = &child;
  root.depth = depth;

  return root;
}


void free_tree_hodlr(struct TreeHODLR *hodlr) {
  struct HODLRNode *node = hodlr->child;

  free(node->data.diagonal.data);

  //node = node->next_sibling;
  free(node->data.off_diagonal.u);
  free(node->data.off_diagonal.v);

  //node = node->next_sibling;
  free(node->data.off_diagonal.u);
  free(node->data.off_diagonal.v);

  //node = node->next_sibling;
  free(node->data.diagonal.data);
}


//int main() {
  //int m = 5, n = 5, leading_dimension = 5;
  //double *matrix;

  //struct TreeHODLR hodlr = dense_to_tree_hodlr(m, n, leading_dimension, matrix);

  // hodlr.data = (double *)malloc(10 * sizeof(double));

  //if (hodlr.data == NULL) {
  ///  printf("Data is null");
  ///}

  //return 0;
//}
