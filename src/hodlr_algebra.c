#include <stdlib.h>
#include <string.h>

#include "../include/tree.h"
#include "../include/error.h"
#include "../include/blas_wrapper.h"
#include "../include/lapack_wrapper.h"


static inline int recompress(
  struct NodeOffDiagonal *restrict const node,
  const int m_larger,
  const int m_smaller,
  const double svd_threshold,
  int *restrict const ierr
) {
  const int nb = node->s < 32 ? node->s : 32;
  int info;
  double *tu = malloc(2 * nb * m_smaller * sizeof(double));
  double *tv = tu + nb * m_smaller;

  double *workspace = malloc(nb * m_larger * sizeof(double));
  dgeqrt_(&node->m, &node->s, &nb, node->u, &node->m, tu, &nb, workspace, &info);
  dgeqrt_(&node->n, &node->s, &nb, node->v, &node->n, tv, &nb, workspace, &info);

  double *r1 = calloc(node->s * node->s, sizeof(double));
  for (int j = 0; j < node->s; j++) {
    for (int i = 0; i < j+1; i++) {
      r1[i + j * node->s] = node->u[i + j * node->m];
    }
  }

  const double alpha = 1.0;
  dtrmm_("R", "U", "T", "N", &node->s, &node->s, &alpha, node->v, &node->n,
         r1, &node->s);

  double *s = malloc(node->s * sizeof(double));
  double *w = malloc(node->s * node->s * sizeof(double));
  double *zt = malloc(node->s * node->s * sizeof(double));

  int result = svd_double(node->s, node->s, node->s, node->s, r1, s, w, zt, ierr);

  int svd_cutoff_idx = 1;
  for (svd_cutoff_idx = 1; svd_cutoff_idx < node->s; svd_cutoff_idx++) {
    if (s[svd_cutoff_idx] < svd_threshold) break;
  }

  double *u_new = calloc(node->m * svd_cutoff_idx, sizeof(double));
  for (int j = 0; j < svd_cutoff_idx; j++) {
    for (int i = 0; i < node->s; i++) {
      u_new[i + j * node->m] = w[i + j * node->s] * s[j];
    }
  }

  dgemqrt_("L", "N", &node->m, &svd_cutoff_idx, &node->s, &nb, node->u, 
           &node->m, tu, &nb, u_new, &node->m, workspace, ierr);
  free(node->u);
  node->u = u_new;

  double *v_new = calloc(node->n * svd_cutoff_idx, sizeof(double));
  for (int j = 0; j < svd_cutoff_idx; j++ ) {
    for (int i = 0; i < node->s; i++) {
      v_new[i + j * node->n] = zt[j + i * node->s];
    }
  }

  dgemqrt_("L", "N", &node->n, &svd_cutoff_idx, &node->s, &nb, 
           node->v, &node->n, tv, &nb, v_new, &node->n, workspace, ierr);

  free(node->v);
  node->v = v_new;
  node->s = svd_cutoff_idx;

  // copy w into mxk array with zeros 
  free(tu); free(workspace); free(r1); free(s); free(w); free(zt);

  return result;
}


static inline int compute_workspace_size_s_component(
  const struct HODLRInternalNode *restrict parent,
  const int height,
  int parent_position
) {
  int which_child = 0, s_sum = 0;

  for (int level = height; level > 0; level--) {
    which_child = (parent_position % 2 == 0) ? 1 : 2;
    s_sum += parent->children[which_child].leaf->data.off_diagonal.s;

    parent = parent->parent;
    parent_position /= 2;
  }
  return s_sum;
}


static inline void compute_higher_level_contributions_off_diagonal(
  const int height,
  const int origin_idx,
  int divisor,
  const struct HODLRInternalNode *restrict parent1,
  const struct HODLRInternalNode *restrict parent2,
  struct NodeOffDiagonal *restrict const out_tr,
  struct NodeOffDiagonal *restrict const out_bl,
  int *restrict const offsets,
  double *restrict const workspace,
  int *restrict const offset_utr_vbl_out,
  int *restrict const offset_vtr_ubl_out
) {
  // TODO: Is divisor always == 1?
  int which_child1 = 0, which_child2 = 0;
  int midx = 2, oidx = 0, parent_position = origin_idx;
  const double alpha = 1.0, beta = 0.0;

  // out_tr->m == out_bl->n => offsets can be reused
  int offset_utr_vbl = 0, offset_vtr_ubl = 0;

  for (int level = height; level > 0; level--) {
    if (origin_idx % divisor == 0) {
      offsets[oidx] = 0;
    }
    if (parent_position % 2 == 0) {
      which_child1 = 1; which_child2 = 2;
    } else {
      which_child1 = 2; which_child2 = 1;
    }
    const int s1 = parent1->children[which_child1].leaf->data.off_diagonal.s;
    const int s2 = parent2->children[which_child2].leaf->data.off_diagonal.s;
    const int m = parent1->children[which_child1].leaf->data.off_diagonal.m;
    const int n = parent1->children[which_child1].leaf->data.off_diagonal.n;

    dgemm_("T", "N", &s1, &s2, &n, &alpha,
           parent1->children[which_child1].leaf->data.off_diagonal.v, &n,
           parent2->children[which_child2].leaf->data.off_diagonal.u, &n,
           &beta, workspace, &s1);

    // TOP-RIGHT OUTPUT
    // Low-rank x low-rank = V* (represents V^T, but not actually transposed)
    dgemm_("N", "T", &out_tr->n, &s1, &s2, &alpha,
           parent2->children[which_child2].leaf->data.off_diagonal.v 
           + (offsets[oidx] + out_bl->n), 
           &m, 
           workspace, &s2, &beta, out_tr->v + offset_vtr_ubl, &out_tr->n);
    dlacpy_("A", &out_tr->m, &s1, 
            parent1->children[which_child1].leaf->data.off_diagonal.u
            + offsets[oidx], &m,
            out_tr->u + offset_utr_vbl, &out_tr->m);

    // BOTTOM-LEFT OUTPUT
    // Low-rank x low-rank = V* (represents V^T, but not actually transposed)
    dgemm_("N", "T", &out_bl->n, &s1, &s2, &alpha,
           parent2->children[which_child2].leaf->data.off_diagonal.v 
           + offsets[oidx], 
           &m, 
           workspace, &s2, &beta, out_bl->v + offset_utr_vbl, &out_bl->n);
    dlacpy_("A", &out_bl->m, &s1, 
            parent1->children[which_child1].leaf->data.off_diagonal.u
            + offsets[oidx] + out_tr->m, &m,
            out_bl->u + offset_vtr_ubl, &out_bl->m);

    offsets[oidx] += out_tr->m + out_tr->n;
    parent1 = parent1->parent; parent2 = parent2->parent;
    midx++; oidx++;
    divisor *= 2; parent_position /= 2;
    offset_utr_vbl += out_tr->m * s1; offset_vtr_ubl += out_tr->n * s1;
  }

  *offset_utr_vbl_out = offset_utr_vbl;
  *offset_vtr_ubl_out = offset_vtr_ubl;
}


static inline void set_up_off_diagonal(
  const struct NodeOffDiagonal *restrict const off_diagonal1,
  const struct NodeOffDiagonal *restrict const off_diagonal2,
  struct NodeOffDiagonal *restrict const out,
  const int s_sum
) {
  const int s_total = s_sum + off_diagonal1->s + off_diagonal2->s;
  out->u = malloc(s_total * out->m * sizeof(double));
  out->v = malloc(s_total * out->n * sizeof(double));
  out->s = s_total;
}


static inline void compute_inner_off_diagonal_lowest_level(
  const struct NodeDiagonal *const diagonal_left,
  const struct NodeOffDiagonal *const off_diagonal_left,
  const struct NodeDiagonal *const diagonal_right,
  const struct NodeOffDiagonal *const off_diagonal_right,
  struct NodeOffDiagonal *restrict const out,
  int offset_u,
  int offset_v
) {
  const double alpha = 1.0, beta = 0.0;

  // Dense x U = U* at index=0
  dgemm_(
    "N", "N", &diagonal_left->m, &off_diagonal_right->s, &diagonal_left->m, 
    &alpha, diagonal_left->data, &diagonal_left->m, 
    off_diagonal_right->u, &off_diagonal_right->m,
    &beta, out->u + offset_u, &diagonal_left->m
  );
  // Copy V
  memcpy(out->v + offset_v, off_diagonal_right->v, 
         off_diagonal_right->n * off_diagonal_right->s * sizeof(double));

  offset_u += diagonal_left->m * off_diagonal_right->s;
  offset_v += off_diagonal_right->s * diagonal_right->m;

  // Copy U
  memcpy(out->u + offset_u, off_diagonal_left->u, 
         off_diagonal_left->m * off_diagonal_left->s * sizeof(double));
  // dense^T x V = V* at index=1 (represents V^T* but not actually transposed)
  dgemm_(
    "T", "N", &diagonal_right->m, &off_diagonal_left->s, &diagonal_right->m, 
    &alpha, diagonal_right->data, &diagonal_right->m, 
    off_diagonal_left->v, &off_diagonal_left->n, &beta, 
    out->v + offset_v, &diagonal_right->m
  );
}


static inline void compute_inner_off_diagonal(
  const int height,
  int parent_position,
  const struct HODLRInternalNode *restrict const parent1,
  const struct HODLRInternalNode *restrict const parent2,
  struct NodeOffDiagonal *restrict const out_tr,
  struct NodeOffDiagonal *restrict const out_bl,
  const double svd_threshold,
  int *restrict offsets,
  double *restrict workspace,
  int *restrict ierr
) {
  const int s_sum = compute_workspace_size_s_component(
    parent1, height, parent_position
  );
  
  set_up_off_diagonal(
    &parent1->children[1].leaf->data.off_diagonal,
    &parent2->children[1].leaf->data.off_diagonal,
    out_tr, s_sum
  );
   
  set_up_off_diagonal(
    &parent1->children[2].leaf->data.off_diagonal,
    &parent2->children[2].leaf->data.off_diagonal,
    out_bl, s_sum
  );

  int offset_utr_vbl, offset_vtr_ubl;
  compute_higher_level_contributions_off_diagonal(
    height-1, parent_position, 1, parent1->parent, parent2->parent, 
    out_tr, out_bl, offsets, workspace, &offset_utr_vbl, &offset_vtr_ubl
  );

  compute_inner_off_diagonal_lowest_level(
    &parent1->children[0].leaf->data.diagonal, 
    &parent1->children[1].leaf->data.off_diagonal,
    &parent2->children[3].leaf->data.diagonal,
    &parent2->children[1].leaf->data.off_diagonal,
    out_tr, offset_utr_vbl, offset_vtr_ubl
  );
   
  compute_inner_off_diagonal_lowest_level(
    &parent1->children[3].leaf->data.diagonal, 
    &parent1->children[2].leaf->data.off_diagonal,
    &parent2->children[0].leaf->data.diagonal,
    &parent2->children[2].leaf->data.off_diagonal,
    out_bl, offset_vtr_ubl, offset_utr_vbl
  );

  int m_larger, m_smaller;
  if (out_tr->m > out_tr->n) {
    m_larger = out_tr->m; m_smaller = out_tr->n;
  } else {
    m_larger = out_tr->n; m_smaller = out_tr->m;
  }
  recompress(out_tr, m_larger, m_smaller, svd_threshold, ierr);
  recompress(out_bl, m_larger, m_smaller, svd_threshold, ierr);
}


static inline void compute_other_off_diagonal_lowest_level(
  const int height,
  const struct HODLRInternalNode *restrict const hodlr1,
  const struct NodeOffDiagonal *restrict const off_diagonal1,
  const struct HODLRInternalNode *restrict const hodlr2,
  const struct NodeOffDiagonal *restrict const off_diagonal2,
  struct NodeOffDiagonal *restrict const out,
  int offset_u,
  int offset_v,
  double *restrict const workspace,
  struct HODLRInternalNode *restrict *restrict queue
) {
  // HODLR x U = U* at index=0
  multiply_internal_node_dense(
    hodlr1, height, off_diagonal2->u, off_diagonal2->s,
    off_diagonal2->m, queue, workspace, out->u + offset_u, off_diagonal2->m
  );
  dlacpy_("A", &off_diagonal2->n, &off_diagonal2->s, off_diagonal2->v, 
          &off_diagonal2->n, out->v + offset_v, &out->m);

  offset_u += hodlr1->m * off_diagonal2->s;
  offset_v += off_diagonal2->s * off_diagonal2->n;

  dlacpy_("A", &off_diagonal1->m, &off_diagonal1->s, off_diagonal1->u,
          &off_diagonal1->m, out->u + offset_u, &out->m);
  // V^T x HODLR = V* at index=1 (represents V^T* but not actually transposed)
  multiply_internal_node_transpose_dense(
    hodlr2, height, off_diagonal1->v, off_diagonal1->s,
    off_diagonal1->m, queue, workspace, out->v + offset_v, off_diagonal1->m
  );
}


static inline void compute_other_off_diagonal(
  const int height,
  const int current_level,
  int parent_position,
  const struct HODLRInternalNode *restrict const parent1,
  const struct HODLRInternalNode *restrict const parent2,
  struct NodeOffDiagonal *restrict out_tr,
  struct NodeOffDiagonal *restrict out_bl,
  struct HODLRInternalNode *restrict *restrict queue,
  int *restrict offsets,
  double *restrict workspace
) {
  const int s_sum = compute_workspace_size_s_component(
    parent1, current_level+1, parent_position
  );

  set_up_off_diagonal(
    &parent1->children[1].leaf->data.off_diagonal,
    &parent2->children[1].leaf->data.off_diagonal,
    out_tr, s_sum
  );
   
  set_up_off_diagonal(
    &parent1->children[2].leaf->data.off_diagonal,
    &parent2->children[2].leaf->data.off_diagonal,
    out_bl, s_sum
  );

  int offset_utr_vbl, offset_vtr_ubl;
  compute_higher_level_contributions_off_diagonal(
    current_level, parent_position, 1, parent1->parent, parent2->parent, 
    out_tr, out_bl, offsets, workspace, &offset_utr_vbl, &offset_vtr_ubl
  );

  compute_other_off_diagonal_lowest_level(
    height - current_level - 1,
    parent1->children[0].internal,
    &parent1->children[1].leaf->data.off_diagonal,
    parent2->children[3].internal,
    &parent2->children[1].leaf->data.off_diagonal,
    out_tr, offset_utr_vbl, offset_vtr_ubl, workspace, queue
  );

  compute_other_off_diagonal_lowest_level(
    height - current_level - 1,
    parent1->children[3].internal,
    &parent1->children[2].leaf->data.off_diagonal,
    parent2->children[0].internal,
    &parent2->children[2].leaf->data.off_diagonal,
    out_tr, offset_utr_vbl, offset_vtr_ubl, workspace, queue
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
  const double svd_threshold,
  int *restrict ierr
) {
  double *workspace, *workspace2;
  int *offsets = calloc(2 * hodlr1->height, sizeof(int));
  if (offsets == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return;
  }
  int *offsets2 = offsets + hodlr1->height;

  //double **matrices = malloc((out->height + 1) * sizeof(double *));

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

    compute_inner_off_diagonal(
      out->height, parent, q1[parent], q2[parent],
      &queue[parent]->children[1].leaf->data.off_diagonal,
      &queue[parent]->children[2].leaf->data.off_diagonal,
      svd_threshold, offsets, workspace, ierr
    );

    queue[parent / 2] = queue[parent]->parent;
    q1[parent / 2] = q1[parent]->parent;
    q2[parent / 2] = q2[parent]->parent;
  }

  for (int level = out->height - 2; level > -1; level--) {
    n_parent_nodes /= 2;

    for (int parent = 0; parent < n_parent_nodes; parent++) {
      compute_other_off_diagonal(
        out->height, level, parent, q1[parent], q2[parent],
        &queue[parent]->children[1].leaf->data.off_diagonal,
        &queue[parent]->children[2].leaf->data.off_diagonal,
        extra_queue, offsets, workspace
      );

      queue[parent / 2] = queue[parent]->parent;
      q1[parent / 2] = q1[parent]->parent;
      q2[parent / 2] = q2[parent]->parent;
    }
  }
  
  free(offsets); free(extra_queue);
}

