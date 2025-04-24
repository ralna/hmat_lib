#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/tree.h"
#include "../include/error.h"
#include "../include/blas_wrapper.h"


static inline void multiply_off_diagonal_vector(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict vector,
  double *restrict out,
  double *restrict workspace,
  double *restrict workspace2,
  const double alpha,
  const double beta,
  const int increment,
  int *restrict offset_ptr,
  const int offset2
) {
  int i = 0;
  int m = parent->children[1].leaf->data.off_diagonal.m;
  int n = parent->children[1].leaf->data.off_diagonal.n;
  int s = parent->children[1].leaf->data.off_diagonal.s;
  
  *offset_ptr += m;
  int offset = *offset_ptr;

  dgemv_("T", &n, &s, &alpha, 
          parent->children[1].leaf->data.off_diagonal.v, 
          &n, vector + offset, &increment, 
          &beta, workspace, &increment);

  dgemv_("N", &m, &s, &alpha, 
          parent->children[1].leaf->data.off_diagonal.u, 
          &m, workspace, &increment,
          &beta, workspace2, &increment);

  for (i = 0; i < m; i++) {
    out[offset2 + i] += workspace2[i];
  }
  
  s = parent->children[2].leaf->data.off_diagonal.s;
  dgemv_("T", &m, &s, &alpha, 
          parent->children[2].leaf->data.off_diagonal.v, 
          &m, vector + offset2, &increment, 
          &beta, workspace, &increment);

  dgemv_("N", &n, &s, &alpha, 
          parent->children[2].leaf->data.off_diagonal.u, 
          &n, workspace, &increment,
          &beta, workspace2, &increment);

  for (i = 0; i < n; i++) {
    out[offset + i] += workspace2[i];
  }
  *offset_ptr += n;
}


double * multiply_vector(const struct TreeHODLR *hodlr,
                         const double *vector,
                         double *out) {
  if (hodlr == NULL) {
    return NULL;
  }
  if (out == NULL) {
    out = malloc(hodlr->root->m * sizeof(double));
  }

  int offset = 0, offset2 = 0, i=0, j=0, k=0, idx=0;
  int m = 0;
  int n_parent_nodes = (int)pow(2, hodlr->height - 1);
  const int increment = 1;
  const double alpha = 1, beta = 0;

  const int largest_m = hodlr->root->children[1].leaf->data.off_diagonal.m ;
  double *workspace = malloc(2 * largest_m * sizeof(double));
  double *workspace2 = workspace + largest_m;

  struct HODLRInternalNode **queue = malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      idx = 2 * i + j;
      m = hodlr->innermost_leaves[idx]->data.diagonal.m;
      dgemv_("N", &m, &m, &alpha, 
             hodlr->innermost_leaves[idx]->data.diagonal.data, 
             &m, vector + offset, &increment, 
             &beta, out + offset, &increment);
      offset += m;
    }
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0; offset2 = 0;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        multiply_off_diagonal_vector(
          queue[idx], vector, out, workspace, workspace2, 
          alpha, beta, increment, &offset, offset2
        );
        offset2 = offset;

        idx += 1;
      }

      queue[j] = queue[2 * j + 1]->parent;
    }
  }

  offset = 0; offset2 = 0;
  multiply_off_diagonal_vector(
    hodlr->root, vector, out, workspace, workspace2, 
    alpha, beta, increment, &offset, offset2
  );

  free(workspace); free(queue);
        
  return out;
}


