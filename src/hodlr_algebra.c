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


static inline void compute_higher_level_contributions_off_diagonal(
  const struct HODLRInternalNode *restrict parent1,
  const struct HODLRInternalNode *restrict parent2,
  double *restrict *restrict matrices,
  double *restrict workspace,
  int *restrict offsets,
  const int height,
  int parent_position,
  const int v_n
) {
  int which_child1 = 0, which_child2 = 0;
  int midx = 2, oidx = 0;
  const double alpha = 1.0, beta = 0.0;

  for (int level = height-1; level > 0; level--) {
    if (parent_position % 2 == 0) {
      which_child1 = 1; which_child2 = 2;
      offsets[oidx] = 0;
    } else {
      which_child1 = 2; which_child2 = 1;
    }
    int s1 = parent1->children[which_child1].leaf->data.off_diagonal.s;
    int s2 = parent1->children[which_child2].leaf->data.off_diagonal.s;
    int m = parent1->children[which_child1].leaf->data.off_diagonal.m;
    int n = parent1->children[which_child1].leaf->data.off_diagonal.n;

    matrices[midx] = malloc(s1 * v_n * sizeof(double));

    // Low-rank x low-rank = V* (represents V^T, but not actually transposed)
    dgemm_("T", "N", &s1, &s2, &n, &alpha,
           parent1->children[which_child1].leaf->data.off_diagonal.v, &s1,
           parent2->children[which_child2].leaf->data.off_diagonal.u, &n,
           &beta, workspace, &s1);
    dgemm_("N", "T", &v_n, &s1, &s2, &alpha,
           parent2->children[which_child2].leaf->data.off_diagonal.v + offsets[oidx],
           &m, workspace, &s2, &beta, matrices[midx], &v_n);

    offsets[oidx] += m;
    parent1 = parent1->parent; parent2 = parent2->parent;
    midx++;
    parent_position /= 2;
  }


}


static inline void compute_inner_off_diagonal(
  const struct HODLRInternalNode *restrict const node1,
  const struct NodeDiagonal *restrict const diagonal1,
  const struct NodeOffDiagonal *restrict const off_diagonal1,
  const struct HODLRInternalNode *restrict const node2,
  const struct NodeDiagonal *restrict const diagonal2,
  const struct NodeOffDiagonal *restrict const off_diagonal2,
  double *restrict workspace,
  double *restrict *restrict matrices,
  struct NodeOffDiagonal *restrict out,
  const int height,
  int parent_position,
  int *restrict offsets
) {
  const double alpha = 1.0, beta = 0.0;

  matrices[0] = malloc((diagonal1->m * off_diagonal2->s + 
                       diagonal2->m * off_diagonal1->s) * sizeof(double));
  matrices[1] = matrices[0] + (diagonal1->m * off_diagonal2->s);

  // Dense x U = U* at index=0
  dgemm_("N", "N", &diagonal1->m, &off_diagonal2->s, &diagonal1->m, &alpha,
         diagonal1->data, &diagonal1->m, off_diagonal2->u, &off_diagonal2->m,
         &beta, matrices[0], &diagonal1->m);

  // V^T x dense = V^T* at index=1 (actually transposed)
  dgemm_("T", "N", &off_diagonal1->s, &diagonal2->m, &off_diagonal1->n, &alpha,
         off_diagonal1->v, &off_diagonal1->s, diagonal2->data, &diagonal2->m,
         &beta, matrices[1], &diagonal2->m);

  compute_higher_level_contributions_off_diagonal(
    node1->parent, node2->parent, matrices, workspace, offsets, height,
    parent_position, off_diagonal1->n
  );

  for (int i = 0; i < height + 1; i++) {
    free(matrices[i]);
  }
}


static inline void compute_other_off_diagonal(
  const struct HODLRInternalNode *restrict const parent1,
  const struct HODLRInternalNode *restrict const hodlr1,
  const struct NodeOffDiagonal *restrict const off_diagonal1,
  const struct HODLRInternalNode *restrict const parent2,
  const struct HODLRInternalNode *restrict const hodlr2,
  const struct NodeOffDiagonal *restrict const off_diagonal2,
  double *restrict workspace,
  double *restrict workspace2,
  double *restrict *restrict matrices,
  struct NodeOffDiagonal *restrict out,
  struct HODLRInternalNode *restrict *restrict queue,
  const int height,
  const int current_level,
  int parent_position,
  int *restrict offsets
) {
  matrices[0] = malloc((hodlr1->m * off_diagonal2->s + 
                        hodlr2->m * off_diagonal1->s) * sizeof(double));
  matrices[1] = matrices[0] + (hodlr1->m * off_diagonal2->s);

  // HODLR x U = U* at index=0
  multiply_internal_node_dense(
    hodlr1, height - current_level - 1, off_diagonal2->u, off_diagonal2->s,
    off_diagonal2->m, queue, workspace, workspace2, matrices[0], 
    off_diagonal2->m
  );

  // V^T x HODLR = V* at index=1 (represents V^T* but not actually transposed)
  multiply_internal_node_transpose_dense(
    hodlr2, height - current_level - 1, off_diagonal1->v, off_diagonal1->s,
    off_diagonal1->m, queue, workspace, workspace2, matrices[1], 
    off_diagonal1->m
  );

  compute_higher_level_contributions_off_diagonal(
    parent1->parent, parent2->parent, matrices, workspace, offsets, 
    current_level, parent_position, off_diagonal1->n
  );

  for (int i = 0; i < height + 1; i++) {
    free(matrices[i]);
  }
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


static void compute_diagonal(
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

    for (int _diagonal = 0; _diagonal < 2; _diagonal++) {
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
      for (int level = out->height; level > 0; level--) {
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

  double **matrices = malloc((out->height + 1) * sizeof(double *));

  struct HODLRInternalNode **queue = out->work_queue;
  struct HODLRInternalNode **q1 = hodlr1->work_queue;
  struct HODLRInternalNode **q2 = hodlr2->work_queue;
  struct HODLRInternalNode **extra_queue =
    malloc(out->len_work_queue * sizeof(struct HODLRInternalNode *));

  compute_diagonal(hodlr1, hodlr2, out, queue, offsets, workspace, 
                   workspace2, ierr);

  long n_parent_nodes = out->len_work_queue;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    q1[parent] = hodlr1->innermost_leaves[2 * parent]->parent;
    q2[parent] = hodlr2->innermost_leaves[2 * parent]->parent;

    // NOTE: The order of these functions matters! (for offsets)
    compute_inner_off_diagonal(
      q1[parent], 
      &q1[parent]->children[3].leaf->data.diagonal, 
      &q1[parent]->children[2].leaf->data.off_diagonal,
      q2[parent], 
      &q2[parent]->children[0].leaf->data.diagonal, 
      &q2[parent]->children[2].leaf->data.off_diagonal,
      workspace, matrices,
      &queue[parent]->children[2].leaf->data.off_diagonal, 
      out->height, parent, offsets
    );
    compute_inner_off_diagonal(
      q1[parent], 
      &q1[parent]->children[0].leaf->data.diagonal, 
      &q1[parent]->children[1].leaf->data.off_diagonal,
      q2[parent], 
      &q2[parent]->children[3].leaf->data.diagonal, 
      &q2[parent]->children[1].leaf->data.off_diagonal,
      workspace, matrices,
      &queue[parent]->children[1].leaf->data.off_diagonal, 
      out->height, parent, offsets
    );

    queue[parent / 2] = queue[parent]->parent;
    q1[parent / 2] = q1[parent]->parent;
    q2[parent / 2] = q2[parent]->parent;
  }

  for (int level = out->height - 2; level > -1; level--) {
    n_parent_nodes /= 2;

    for (int parent = 0; parent < n_parent_nodes; parent++) {
      // NOTE: The order here is again important!
      compute_other_off_diagonal(
        q1[parent], 
        q1[parent]->children[3].internal, 
        &q1[parent]->children[2].leaf->data.off_diagonal,
        q2[parent],
        q2[parent]->children[0].internal,
        &q2[parent]->children[2].leaf->data.off_diagonal,
        workspace, workspace2, matrices, 
        &queue[parent]->children[2].leaf->data.off_diagonal,
        extra_queue, out->height, level, parent, offsets
      );
      compute_other_off_diagonal(
        q1[parent], 
        q1[parent]->children[0].internal, 
        &q1[parent]->children[1].leaf->data.off_diagonal,
        q2[parent],
        q2[parent]->children[3].internal,
        &q2[parent]->children[1].leaf->data.off_diagonal,
        workspace, workspace2, matrices, 
        &queue[parent]->children[1].leaf->data.off_diagonal,
        extra_queue, out->height, level, parent, offsets
      );

      queue[parent / 2] = queue[parent]->parent;
      q1[parent / 2] = q1[parent]->parent;
      q2[parent / 2] = q2[parent]->parent;
    }
  }
  
  free(offsets); free(extra_queue);
}

