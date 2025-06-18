#include <stdlib.h>

#include "../include/tree.h"
#include "../include/error.h"
#include "../include/blas_wrapper.h"


static void compute_off_diagonal(
  const struct HODLRInternalNode *restrict const internal1,
  const int height1,
  const struct HODLRLeafNode *restrict const leaf1,
  const struct HODLRInternalNode *restrict const internal2,
  const int height2,
  const struct HODLRLeafNode *restrict const leaf2,
  const struct HODLRInternalNode **restrict queue,
  double *restrict workspace,
  double *restrict workspace2,
  struct HODLRLeafNode *restrict out,
  int *ierr
) {
  double *u = malloc(internal1->m * leaf2->data.off_diagonal.s * sizeof(double));
  if (u == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return;
  }

  multiply_internal_node_dense(
    internal1, height1, leaf2->data.off_diagonal.u, leaf2->data.off_diagonal.s,
    leaf2->data.off_diagonal.m, queue, workspace, workspace2, u, 
    leaf2->data.off_diagonal.m
  );

  double *v = malloc(leaf1->data.off_diagonal.s * internal2->m * sizeof(double));
  if (v == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return;
  }

  multiply_internal_node_transpose_dense(
    internal2, height2, leaf1->data.off_diagonal.v, leaf1->data.off_diagonal.s,
     leaf2->data.off_diagonal.n, queue, workspace, workspace2, v, 
    leaf2->data.off_diagonal.n
  );
 
}


static inline void add_off_diagonal_contribution(
  const struct NodeOffDiagonal *restrict const leaf1,
  const struct NodeOffDiagonal *restrict const leaf2,
  const int offset,
  double *restrict workspace,
  double *restrict workspace2,
  double *restrict out,
  const int m
) {
  const double alpha = 1.0, beta = 0.0;

  dgemm_("T", "N", &leaf1->s, &leaf2->s, &leaf1->n, &alpha,
         leaf1->v, &leaf1->n, leaf2->u, &leaf2->m,
         &beta, workspace, &leaf1->s);

  dgemm_("N", "T", &leaf1->s, &m, &leaf2->s, &alpha,
         workspace, &leaf1->s, leaf2->v + offset, &leaf2->s,
         &beta, workspace2, &leaf1->s);

  dgemm_("N", "N", &m, &m, &leaf1->s, &alpha,
         leaf1->u + offset, &leaf1->m, workspace2, &leaf1->s, &alpha, 
         out, &m);
}


void compute_diagonal(
  const struct TreeHODLR *restrict const hodlr1,
  const struct TreeHODLR *restrict const hodlr2,
  struct TreeHODLR *restrict out,
  struct HODLRInternalNode **restrict queue,
  int *restrict offsets,
  double *restrict workspace,
  double *restrict workspace2,
  int *restrict ierr
) {
  struct HODLRInternalNode *parent_node1 = NULL, *parent_node2 = NULL;
  int m = 0, idx = 0, oidx = 0;
  const double alpha = 1.0, beta = 0.0;
  int which_child1 = 0, which_child2 = 0;

  for (int parent = 0; parent < out->len_work_queue; parent++) {
    queue[parent] = out->innermost_leaves[2 * parent]->parent;

    for (int child = 0; child < 2; child++) {
      m = out->innermost_leaves[idx]->data.diagonal.m;
      out->innermost_leaves[idx]->data.diagonal.data = 
        malloc(m * m * sizeof(double));
      if (out->innermost_leaves[idx]->data.diagonal.data == NULL) {
        *ierr = ALLOCATION_FAILURE;
        return;
      }

      dgemm_(
        "N", "N", &m, &m, &m, &alpha,
        hodlr1->innermost_leaves[idx]->data.diagonal.data, &m,
        hodlr2->innermost_leaves[idx]->data.diagonal.data, &m,
        &beta, out->innermost_leaves[idx]->data.diagonal.data, &m
      );

      parent_node1 = hodlr1->innermost_leaves[idx]->parent;
      parent_node2 = hodlr2->innermost_leaves[idx]->parent;

      int position = idx;
      oidx = 0;
      for (int level = out->height; level > 0; level++) {
        if (position % 2 == 0) {
          which_child1 = 1; which_child2 = 2;
          offsets[oidx] = 0;
        } else {
          which_child1 = 2; which_child2 = 1;
        }
        add_off_diagonal_contribution(
          &parent_node1->children[which_child1].leaf->data.off_diagonal,
          &parent_node2->children[which_child2].leaf->data.off_diagonal,
          offsets[oidx], workspace, workspace2, 
          out->innermost_leaves[idx]->data.diagonal.data, 
          out->innermost_leaves[idx]->data.diagonal.m
        );

        offsets[oidx] += out->innermost_leaves[idx]->data.diagonal.m;

        parent_node1 = parent_node1->parent;
        parent_node2 = parent_node2->parent;
        position /= 2;
        oidx++;
      }

      idx++;
    }
  }
}


void multiply_hodlr_hodlr(
  const struct TreeHODLR *restrict const hodlr1,
  const struct TreeHODLR *restrict const hodlr2,
  struct TreeHODLR *restrict out,
  int *restrict ierr
) {
  double *workspace, *workspace2;
  int *offsets = calloc(hodlr1->height, sizeof(int));
  if (offsets == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return;
  }

  struct HODLRInternalNode **queue = out->work_queue;

  compute_diagonal(hodlr1, hodlr2, out, queue, offsets, workspace, 
                   workspace2, ierr);
  
  free(offsets);
}

