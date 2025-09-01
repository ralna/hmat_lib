#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/hmat_lib/hodlr.h"
#include "../include/hmat_lib/allocators.h"
#include "../include/hmat_lib/constructors.h"
#include "../include/hmat_lib/vector_algebra.h"
#include "../include/hmat_lib/dense_algebra.h"

#include "../tests/include/io.h"
#include "../include/internal/blas_wrapper.h"


#define MAX_PRINT_M 25


static void print_matrix(int m, int n, double *matrix) {
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      printf("%f    ", matrix[j * m + i]);
    }
    printf("\n");
  }
  printf("\n");
}


static void print_vector(int m, double *vector) {
  for (int i = 0; i < m; i++) {
    printf("%f    ", vector[i]);
  }
  printf("\n");
}


static void print_node_diagonal(struct NodeDiagonal *node) {
  int m = node->m;

  printf("(%dx%d) node=%p data=%p\n", m, m, node, node->data);

  if (m < MAX_PRINT_M) {
    for (int i=0; i<m; i++) {
      for (int j=0; j < m; j++) {
        printf("%f    ", node->data[j * m + i]);
      }
      printf("\n");
    }
  }
  printf("\n");
}


static void print_node_offdiagonal(struct NodeOffDiagonal *node) {
  int m = node->m; int s = node->s; int n = node->n;

  printf("U (%dx%d):\n", m, s);
  if (m < MAX_PRINT_M) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < s; j++) {
        printf("%f    ", node->u[j * m + i]);
      }
      printf("\n");
    }
  }

  printf("\nV_T (%dx%d):\n", s, n);
  if (n < MAX_PRINT_M) {
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < n; j++) {
        printf("%f    ", node->v[j + i * n]);
      }
      printf("\n");
    }
  }
  printf("\n");
}


static void print_tree_hodlr(struct TreeHODLR *hodlr) {
  int len_queue = 1;
  int max_depth_n = (int)pow(2, hodlr->height - 1); 

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **temp_pointer;
  queue[0] = hodlr->root;

  for (int i = 1; i < hodlr->height; i++) {
    printf("depth=%d\n", i-1);
    for (int j = 0; j < len_queue; j++) {
      printf("node=%d\nTOP RIGHT CORNER:\n", j);
      print_node_offdiagonal(&queue[j]->children[1].leaf->data.off_diagonal);

      printf("BOTTOM LEFT CORNER:\n");
      print_node_offdiagonal(&queue[j]->children[2].leaf->data.off_diagonal);
    
      next_level[2 * j] = queue[j]->children[0].internal;
      next_level[2 * j + 1] = queue[j]->children[3].internal;
    }
    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;

    len_queue *= 2;
  }

  printf("depth=MAX\n");
  for (int i = 0; i < len_queue; i++) {
    printf("TOP LEFT CORNER:\n");
    print_node_diagonal(&queue[i]->children[0].leaf->data.diagonal);

    printf("TOP RIGHT CORNER:\n");
    print_node_offdiagonal(&queue[i]->children[1].leaf->data.off_diagonal);
    
    printf("BOTTOM LEFT CORNER:\n");
    print_node_offdiagonal(&queue[i]->children[2].leaf->data.off_diagonal);
    
    printf("BOTTOM RIGHT CORNER:\n");
    print_node_diagonal(&queue[i]->children[3].leaf->data.diagonal);
  }

  free(queue); free(next_level);
}


void construct_laplacian_matrix(int m, double *matrix) {
  int idx = 0;
  for (int i = 0; i<m; i++) {
    for (int j = 0; j<m; j++) {
      idx = j + i * m;
      if (i == j) {
        matrix[idx] = 2;
      } else if (i == j+1 || i == j-1) {
        matrix[idx] = -1;
      } else {
        matrix[idx] = 0;
      }
    }
  }
}


int main(int argc, char **argv) {
  int m = 21;
  double svd_threshold = 1e-8;
  int depth = 3, ierr;
  int ms[8] = {1, 4, 3, 2, 2, 4, 4, 1};

  int idx; double *matrix;

  if (argc == 1) {
    matrix = malloc(m * m * sizeof(double));
    construct_laplacian_matrix(m, matrix);
    //matrix[m - 1] = 0.5;
    //matrix[m * (m - 1)] = 0.5;
    print_matrix(m, m, matrix);
  } else if (argc == 2) {
    matrix = read_dense_matrix(argv[1], &m, &malloc, &free);
    if (matrix == NULL) {
      printf("Matrix from '%s' could not be read, aborting...\n", argv[1]);
    }
  } else {
    printf("Incorrect number of arguments (%d)\n", argc);
    return 1;
  }

  double *matrix2 = malloc(m * m * sizeof(double));
  memcpy(matrix2, matrix, m * m * sizeof(double));

  printf("%d x %d matrix initialised - constructing HOLDR matrix...\n", m, m);
  struct TreeHODLR *hodlr = allocate_tree(depth, &ierr);
  printf("HODLR matrix allocated, converting from dense...\n");

  dense_to_tree_hodlr(hodlr, m, &ms[0], matrix, svd_threshold, &ierr);
  printf("HODLR matrix computed, printing...\n");

  print_tree_hodlr(hodlr);

  double *vector = malloc(m * sizeof(double));
  for (int i = 0; i < m; i++) {
    vector[i] = (double)rand() / RAND_MAX;
  }

  printf("HODLR vector multiplication:\n");
  double *result = multiply_vector(hodlr, vector, NULL, &ierr);
  if (m < 4 * MAX_PRINT_M) {
    print_vector(m, result);
  }

  printf("Computing reference using blas...\n");
  double *result2 = malloc(m * sizeof(double));
  const double alpha = 1.0, beta = 0.0;
  const int incx = 1;
  dgemv_("N", &m, &m, &alpha, matrix2, &m, vector, &incx, &beta, result2, &incx);

  printf("Comparing HODLR mult with BLAS:\n");
  double normv = 0.0, diff = 0.0;
  for (int i = 0; i < m; i++) {
    normv += result[i] * result[i];
    diff += (result[i] - result2[i]) * (result[i] - result2[i]);
  }
  printf("normv=%f, diff=%f, relerr=%f\n\n", sqrt(normv), sqrtf(diff), 
         sqrtf(diff) / sqrtf(normv));
  free(result2); free(result);
  /* for (int i = 0; i < m; i++) { */
  /*   for (int j = 0 ; j < m; j++) { */
  /*     if (i == j) { */
  /*       matrix[i + j * m] = 1.0; */
  /*     } else { */
  /*       matrix[i + j * m] = 0.0; */
  /*     } */
  /*   } */
  /* } */

  construct_laplacian_matrix(m, matrix);
  //printf("\n\n");
  printf("\n\nHODLR dense matrix multiplication:\n");

  result = multiply_hodlr_dense(hodlr, matrix, m, m, NULL, m, &ierr);
  if (m < MAX_PRINT_M) {
    print_matrix(m, m, result);
  }
  free(result);

  result = multiply_dense_hodlr(hodlr, matrix, m, m, NULL, m, &ierr);
  if (m < MAX_PRINT_M) {
    print_matrix(m, m, result);
  }
  free(result);

  // Always free:
  free_tree_hodlr(&hodlr);
  free(matrix);

  free(vector); free(matrix2);

  return 0;
}
