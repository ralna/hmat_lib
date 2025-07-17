#include <stdlib.h>
#include <stdio.h>

#include "../include/tree.h"
#include "../include/error.h"
#include "../include/blas_wrapper.h"


static inline void recompress(
  double *restrict *restrict us,
  double *restrict *restrict vs,
  struct NodeOffDiagonal *restrict out
) {

}


static inline void compute_uv_workspace_size(
  const struct HODLRInternalNode *restrict parent1,
  const struct HODLRInternalNode *restrict parent2,
  int *restrict const offsets,
  const int height,
  int parent_position,
  const int target_m,
  const int target_n,
  int *restrict const u_size_out,
  int *restrict const v_size_out
) {
  int which_child1 = 0, which_child2 = 0, u_size = 0, v_size = 0;

  for (int level = height-1; level > 0; level--) {
    if (parent_position % 2 == 0) {
      which_child1 = 1; which_child2 = 2;
    } else {
      which_child1 = 2; which_child2 = 1;
    }
    int s1 = parent1->children[which_child1].leaf->data.off_diagonal.s;

    u_size += s1 * target_m;
    v_size += s1 * target_n;

    parent1 = parent1->parent; parent2 = parent2->parent;
    parent_position /= 2;
  }

  *u_size_out += u_size;
  *v_size_out += v_size;
}


static inline void compute_higher_level_contributions_off_diagonal(
  const int height,
  const int origin_idx,
  int divisor,
  const struct HODLRInternalNode *restrict parent1,
  int *restrict const offsets1,
  const struct HODLRInternalNode *restrict parent2,
  int *restrict const offsets2,
  struct NodeOffDiagonal *restrict const out_tr,
  struct NodeOffDiagonal *restrict const out_bl,
  double *restrict const workspace
) {
  int which_child1 = 0, which_child2 = 0;
  int midx = 2, oidx = 0, parent_position = origin_idx;
  const double alpha = 1.0, beta = 0.0;

  // out_tr->m == out_bl->n => offsets can be reused
  int offset_utr_vbl = 0, offset_vtr_ubl = 0;

  for (int level = height-1; level > 0; level--) {
    if (origin_idx % divisor) {
      offsets1[oidx] = 0; offsets2[oidx] = out_bl->n;
    }
    if (parent_position % 2 == 0) {
      which_child1 = 1; which_child2 = 2;
    } else {
      which_child1 = 2; which_child2 = 1;
    }
    int s1 = parent1->children[which_child1].leaf->data.off_diagonal.s;
    int s2 = parent1->children[which_child2].leaf->data.off_diagonal.s;
    int m = parent1->children[which_child1].leaf->data.off_diagonal.m;
    int n = parent1->children[which_child1].leaf->data.off_diagonal.n;

    dgemm_("T", "N", &s1, &s2, &n, &alpha,
           parent1->children[which_child1].leaf->data.off_diagonal.v, &n,
           parent2->children[which_child2].leaf->data.off_diagonal.u, &n,
           &beta, workspace, &s1);

    // TOP-RIGHT OUTPUT
    // Low-rank x low-rank = V* (represents V^T, but not actually transposed)
    dgemm_("N", "T", &out_tr->n, &s1, &s2, &alpha,
           parent2->children[which_child2].leaf->data.off_diagonal.v 
           + offsets2[oidx], &m, 
           workspace, &s2, &beta, out_tr->v + offset_vtr_ubl, &out_tr->n);
    dlacpy_("A", &out_tr->m, &s1, 
            parent1->children[which_child1].leaf->data.off_diagonal.u
            + offsets1[oidx], &m,
            out_tr->u + offset_utr_vbl, &out_tr->m);

    // BOTTOM-LEFT OUTPUT
    // Low-rank x low-rank = V* (represents V^T, but not actually transposed)
    dgemm_("N", "T", &out_bl->n, &s1, &s2, &alpha,
           parent2->children[which_child2].leaf->data.off_diagonal.v 
           + offsets1[oidx], &m, 
           workspace, &s2, &beta, out_bl->v + offset_utr_vbl, &out_bl->n);
    dlacpy_("A", &out_bl->m, &s1, 
            parent1->children[which_child1].leaf->data.off_diagonal.u
            + offsets2[oidx], &m,
            out_bl->u + offset_vtr_ubl, &out_bl->m);

    offsets1[oidx] += out_tr->m + out_bl->m;
    offsets2[oidx] += out_tr->n + out_bl->n;
    parent1 = parent1->parent; parent2 = parent2->parent;
    midx++;
    divisor *=2; parent_position /= 2;
    offset_utr_vbl += out_tr->m * s1; offset_vtr_ubl += out_tr->n * s1;
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
  struct NodeOffDiagonal *restrict out,
  const int height,
  int parent_position,
  int *restrict offsets
) {
  const double alpha = 1.0, beta = 0.0;

  const int u_size1 = diagonal1->m * off_diagonal2->s;
  const int u_size2 = off_diagonal1->m * off_diagonal1->s;
  const int v_size1 = off_diagonal2->s * off_diagonal2->n;
  const int v_size2 = off_diagonal1->s * diagonal2->m;

  int u_size = u_size1 + u_size2, v_size = v_size1 + v_size2;

  compute_uv_workspace_size(
    node1->parent, node2->parent, offsets, height, parent_position, 
    diagonal1->m, diagonal2->m, &u_size, &v_size
  );

  double *u = malloc((u_size + v_size) * sizeof(double));
  double *v = u + u_size;

  // Dense x U = U* at index=0
  dgemm_("N", "N", &diagonal1->m, &off_diagonal2->s, &diagonal1->m, &alpha,
         diagonal1->data, &diagonal1->m, off_diagonal2->u, &off_diagonal2->m,
         &beta, u, &diagonal1->m);
  dlacpy_("A", &off_diagonal2->n, &off_diagonal2->s, off_diagonal2->v,
          &off_diagonal2->n, v, &diagonal1->m);

  // V^T x dense = V* at index=1 (represents V^T* but not actually transposed)
  dlacpy_("A", &off_diagonal1->m, &off_diagonal1->s, off_diagonal1->u,
          &off_diagonal1->m, u + u_size1, &diagonal2->m);
  dgemm_("T", "N", &diagonal2->m, &off_diagonal1->s, &diagonal2->m, &alpha,
         diagonal2->data, &diagonal2->m, off_diagonal1->v, &off_diagonal1->n,
         &beta, v + v_size1, &diagonal2->m);

  compute_higher_level_contributions_off_diagonal(
    height, parent_position, node1->parent, node2->parent, 
    u + u_size1 + u_size2, out->m, v + v_size1 + v_size2, out->n, 
    offsets, workspace
  );

  free(u);
}


static inline void compute_other_off_diagonal(
  const struct HODLRInternalNode *restrict const parent1,
  const struct HODLRInternalNode *restrict const hodlr1,
  const struct NodeOffDiagonal *restrict const off_diagonal1,
  const struct HODLRInternalNode *restrict const parent2,
  const struct HODLRInternalNode *restrict const hodlr2,
  const struct NodeOffDiagonal *restrict const off_diagonal2,
  double *restrict workspace,
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
    off_diagonal2->m, queue, workspace, matrices[0], off_diagonal2->m
  );

  // V^T x HODLR = V* at index=1 (represents V^T* but not actually transposed)
  multiply_internal_node_transpose_dense(
    hodlr2, height - current_level - 1, off_diagonal1->v, off_diagonal1->s,
    off_diagonal1->m, queue, workspace, matrices[1], off_diagonal1->m
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
         workspace, &leaf1->s, leaf2->v + offset, &leaf2->n,
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
      m = hodlr1->innermost_leaves[idx]->data.diagonal.m;

      out->innermost_leaves[idx]->data.diagonal.m = m;
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

      int divisor = 1, position = idx;
      oidx = 0;
      for (int level = out->height; level > 0; level--) {
        if (idx % divisor == 0) {
          offsets[oidx] = 0;
        }
        if (position % 2 == 0) {
          which_child1 = 1; which_child2 = 2;
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

        offsets[oidx] += m;

        parent_node1 = parent_node1->parent;
        parent_node2 = parent_node2->parent;
        divisor *= 2; position /= 2;
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
        workspace, matrices, 
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
        workspace, matrices, 
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

