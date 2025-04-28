#include <stdio.h>
#include <stdlib.h>

#include "../include/lapack_wrapper.h"
#include "../include/tree.h"
#include "../include/error.h"
#include "../include/blas_wrapper.h"


static void print_matrix(const int m, const int n, 
                         const double *matrix, const int lda) {
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      printf("%f    ", matrix[j * lda + i]);
    }
    printf("\n");
  }
  printf("\n");
}


void compute_multiply_hodlr_dense_workspace(
  const struct TreeHODLR *restrict hodlr,
  const int matrix_n,
  int *restrict workspace_sizes
) {
  int i = 0, j = 0, k = 0, idx = 0, s = 0;
  long n_parent_nodes = hodlr->len_work_queue;

  struct HODLRInternalNode **queue = hodlr->work_queue;

  workspace_sizes[1] = hodlr->root->children[1].leaf->data.off_diagonal.m * matrix_n;
  if (hodlr->height < 2) {
    i = hodlr->root->children[1].leaf->data.off_diagonal.s;
    j = hodlr->root->children[2].leaf->data.off_diagonal.s;
    s = i > j ? i : j;
    workspace_sizes[0] = s * matrix_n;
    return;
  }

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        s = queue[idx + k]->children[1].leaf->data.off_diagonal.s;
        if (s > workspace_sizes[0]) {
          workspace_sizes[0] = s;
        }

        s = queue[idx + k]->children[2].leaf->data.off_diagonal.s;
        if (s > workspace_sizes[0]) {
          workspace_sizes[0] = s;
        }
      }
    } 
  }

  workspace_sizes[0] *= matrix_n;
}


static inline void multiply_off_diagonal_dense(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict matrix,
  const int matrix_n,
  const int ldb,
  double *restrict out,
  const int out_ld,
  double *restrict workspace,
  double *restrict workspace2,
  const double alpha,
  const double beta,
  int *restrict offset_ptr,
  const int offset2
) {
  int i = 0, j = 0;
  int m = parent->children[1].leaf->data.off_diagonal.m;
  int n = parent->children[1].leaf->data.off_diagonal.n;
  int s = parent->children[1].leaf->data.off_diagonal.s;
  
  *offset_ptr += m;
  int offset = *offset_ptr;

  print_matrix(m, s, parent->children[1].leaf->data.off_diagonal.u, m);
  dgemm_("T", "N", &s, &matrix_n, &n, &alpha, 
         parent->children[1].leaf->data.off_diagonal.v, 
         &n, matrix + offset, &ldb, 
         &beta, workspace, &s);
  print_matrix(s, matrix_n, workspace, s);

  dgemm_("N", "N", &m, &matrix_n, &s, &alpha, 
         parent->children[1].leaf->data.off_diagonal.u, 
         &m, workspace, &s,
         &beta, workspace2, &m);
  print_matrix(m, matrix_n, workspace2, m);

  for (j = 0; j < matrix_n; j++) {
    for (i = 0; i < m; i++) {
      out[offset2 + i + j * out_ld] += workspace2[i + j * m];
    }
  }
  
  s = parent->children[2].leaf->data.off_diagonal.s;
  dgemm_("T", "N", &s, &matrix_n, &m, &alpha, 
         parent->children[2].leaf->data.off_diagonal.v, 
         &m, matrix + offset2, &ldb, 
         &beta, workspace, &s);
  print_matrix(s, matrix_n, workspace, m);

  dgemm_("N", "N", &n, &matrix_n, &s, &alpha, 
         parent->children[2].leaf->data.off_diagonal.u, 
         &n, workspace, &s,
         &beta, workspace2, &n);
  print_matrix(n, matrix_n, workspace2, n);

  for (j = 0; j < matrix_n; j++) {
    for (i = 0; i < n; i++) {
      out[offset + i + j * out_ld] += workspace2[i + j * n];
    }
  }
  *offset_ptr += n;
}


double * multiply_hodlr_dense(const struct TreeHODLR *hodlr,
                              const double *restrict matrix,
                              const int matrix_n,
                              const int matrix_ld,
                              double *restrict out,
                              const int out_ld) {
  if (hodlr == NULL) {
    return NULL;
  }
  if (out == NULL) {
    out = malloc(out_ld * matrix_n * sizeof(double));
  }

  int offset = 0, offset2 = 0, i=0, j=0, k=0, idx=0;
  int m = 0;
  long n_parent_nodes = hodlr->len_work_queue;
  const double alpha = 1.0, beta = 0.0;

  //print_matrix(hodlr->root->m, matrix_n, matrix, matrix_ld);

  int workspace_size[2] = {0, 0};
  compute_multiply_hodlr_dense_workspace(hodlr, matrix_n, &workspace_size);
  double *workspace = malloc((workspace_size[0] + workspace_size[1]) * sizeof(double));
  double *workspace2 = workspace + workspace_size[0];
  printf("%d %d\n", workspace_size[0], workspace_size[1]);

  struct HODLRInternalNode **queue = hodlr->work_queue;
  
  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      idx = 2 * i + j;
      m = hodlr->innermost_leaves[idx]->data.diagonal.m;
      dgemm_("N", "N", &m, &matrix_n, &m, &alpha, 
             hodlr->innermost_leaves[idx]->data.diagonal.data, 
             &m, matrix + offset, &matrix_ld,
             &beta, out + offset, &out_ld);
      
      offset += m;
    }
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0; offset2 = 0;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        multiply_off_diagonal_dense(
          queue[idx], matrix, matrix_n, matrix_ld, 
          out, out_ld, workspace, workspace2, 
          alpha, beta, &offset, offset2
        );
        offset2 = offset;

        idx += 1;
      }

      queue[j] = queue[2 * j + 1]->parent;
    }
  }

  offset = 0; offset2 = 0;
  multiply_off_diagonal_dense(
    hodlr->root, matrix, matrix_n, matrix_ld, 
    out, out_ld, workspace, workspace2, 
    alpha, beta, &offset, offset2
  );

  free(workspace);
        
  return out;
}


