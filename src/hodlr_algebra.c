#include <stdlib.h>
#include <string.h>

#include "../include/tree.h"
#include "../include/error.h"
#include "../include/utils.h"
#include "../include/blas_wrapper.h"
#include "../include/lapack_wrapper.h"


/**
 * Recompresses an off-diagonal leaf node using the QR method.
 *
 * Should be used when ``node->s < min(node->m, node->n)`` since the method
 * uses an intermediate ``node->s`` x ``node->s`` matrix for the SVD. For 
 * nodes with large ``s``, :c:func:`recompress_large_s` should be used 
 * instead.
 *
 * Parameters
 * ----------
 * node
 *     Pointer to an off-diagonal leaf node to recompress. Must be fully 
 *     constructed with block sizes set and the data containing the 
 *     uncompressed U and V matrices.
 * m_larger
 *     The larger value between ``node->m`` and ``node->n``.
 * m_smaller
 *     The smaller value between ``node->m`` and ``node->n``.
 * svd_threshold
 *     The threshold for discarding singular values when recompressing 
 *     off-diagonal nodes - any singular values smaller than one 
 *     ``svd_threshold``-th of the first singular value will be treated as 
 *     approximately zero and therefore the corresponding column vectors of 
 *     the :math:`U` and :math:`V` matrices will be discarded.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. An error status code from :c:enum:`ErrorCode` is written into 
 *     the pointer **if an error occurs**. Must not be ``NULL`` - doing so is 
 *     undefined.
 *
 * Return
 * ------
 * int
 *     The status code returned by ``dgesdd``.
 *
 * Notes
 * -----
 * The recompression is performed in the following steps:
 *
 * 1. QR decomposition of :math:`U` matrix: :math:`U -> Q_U R_U`
 *
 *    * :math:`Q_U` is stored in the original :math:`U` matrix
 *    * The reflectors for :math:`R_U` are stored in a new ``nb`` x 
 *      ``m_smaller`` matrix.
 *    * The operation also requires a ``nb`` x ``m_larger`` workspace.
 *
 * 2. QR decomposition of :math:`V` matrix: :math:`V -> Q_V R_V`
 *
 *    * :math:`Q_V` is stored in the original :math:`V` matrix
 *    * The reflectors for :math:`R_V` are stored in a new ``nb`` x 
 *      ``m_smaller`` matrix.
 *    * The operation also requires a ``nb`` x ``m_larger`` workspace.
 *
 * 3. Matrix multiplication of the :math:`R` matrices: :math:`R = R_U R_V^T`
 *
 *    * :math:`R` is stored in a new ``node->s`` x ``node->s`` matrix.
 *
 * 4. SVD of the result: :math:`R -> W \Sigma Z^T`
 *
 *    * :math:`W` is stored in a new ``node->s`` x ``node->s`` matrix.
 *    * :math:`\Sigma` is stored in a new ``node->s`` array.
 *    * :math:`Z^T` is stored in a new ``node->s`` x ``node->s`` matrix.
 *
 * 5. Left-side matrix multiply: :math:`U_{new} = Q_U W`
 *
 *    * :math:`U_{new}` is stored in a new ``node->m`` x ``s`` matrix.
 *
 * 6. Right-side matrix multiply: :matrix:`V_{new} = Q_V R_V^T`
 *
 *    * :math:`V_{new}` is stored in a new ``node->n`` x ``s`` matrix.
 */
static inline int recompress(
  struct NodeOffDiagonal *restrict const node,
  const int m_larger,
  const int m_smaller,
  const double svd_threshold,
  int *restrict const ierr
) {
  const int nb = node->s < 32 ? node->s : 32;
  int info;

  const unsigned int t_size = nb * m_smaller;
  const unsigned int w_size = nb * node->s;
  double *mem = 
    malloc((2 * t_size + w_size + 2 * node->s * node->s + node->s) * 
           sizeof(double));

  double *tu = &mem[0];
  double *tv = tu + t_size;
  double *workspace = tv + t_size;

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

  double *s = workspace + w_size;
  double *w = s + node->s;
  double *zt = w + node->s * node->s;

  int result = 
    svd_double(node->s, node->s, node->s, node->s, r1, s, w, zt, ierr);

  int svd_cutoff_idx = 1;
  for (svd_cutoff_idx = 1; svd_cutoff_idx < node->s; svd_cutoff_idx++) {
    if (s[svd_cutoff_idx] <= svd_threshold) break;
  }

  // Compute new U
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

  // Compute new V
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

  free(mem); free(r1);

  return result;
}


/**
 * Recompresses an off-diagonal leaf node using the direct SVD method.
 *
 * Should be used when ``node->s >= min(node->m, node->n)`` since the method
 * uses an intermediate ``node->m`` x ``node->n`` matrix for the SVD. For 
 * nodes with small ``s``, :c:func:`recompress` should be used instead.
 *
 * Parameters
 * ----------
 * node
 *     Pointer to an off-diagonal leaf node to recompress. Must be fully 
 *     constructed with block sizes set and the data containing the 
 *     uncompressed U and V matrices.
 * m_larger
 *     The larger value between ``node->m`` and ``node->n``.
 * m_smaller
 *     The smaller value between ``node->m`` and ``node->n``.
 * svd_threshold
 *     The threshold for discarding singular values when recompressing 
 *     off-diagonal nodes - any singular values smaller than one 
 *     ``svd_threshold``-th of the first singular value will be treated as 
 *     approximately zero and therefore the corresponding column vectors of 
 *     the :math:`U` and :math:`V` matrices will be discarded.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. An error status code from :c:enum:`ErrorCode` is written into 
 *     the pointer **if an error occurs**. Must not be ``NULL`` - doing so is 
 *     undefined.
 *
 * Return
 * ------
 * int
 *     The status code returned by ``dgesdd``.
 *
 * Notes
 * -----
 * Performs the recompression in the following steps:
 *
 * 1. Expand the low-rank matrix: :math:`D = U V^T`
 *
 *    * :math:`D` is stored in a ``node->m`` x ``node->n`` workspace.
 *
 * 2. Perform SVD on the dense matrix: :math:`D -> W \Sigma Z^T`
 *
 *    * :math:`W` is stored in a new ``node->s`` x ``node->s`` matrix.
 *    * :math:`\Sigma` is stored in a new ``node->s`` array.
 *    * :math:`Z^T` is stored in a new ``node->s`` x ``node->s`` matrix.
 *
 * 3. Store the relevant columns of :math:`W` as :math:`U_{new}`
 *
 *    * :math:`U_{new}` is stored in a new ``node->m`` x ``s`` matrix.
 *
 * 4. Store the relevant rows of :math:`Z^T` as :math:`V_{new}`
 *
 *    * :math:`V_{new}` is stored in a new ``node->n`` x ``s`` matrix.
 */
static inline int recompress_large_s(
  struct NodeOffDiagonal *restrict const node,
  const int m_smaller,
  const double svd_threshold,
  int *restrict const ierr
) {
  double *s = malloc((node->m * node->n + m_smaller) * sizeof(double));
  double *workspace = s + m_smaller;

  const double alpha = 1.0, beta = 0.0;
  dgemm_("N", "T", &node->m, &node->n, &node->s, &alpha, node->u, &node->m,
         node->v, &node->n, &beta, workspace, &node->m);

  const int result = svd_double(
    node->m, node->n, m_smaller, node->m, workspace, s, node->u, node->v, ierr
  );

  int svd_cutoff_idx = 1;
  for (svd_cutoff_idx = 1; svd_cutoff_idx < m_smaller; svd_cutoff_idx++) {
    if (s[svd_cutoff_idx] <= svd_threshold) break;
  }

  // Copy new U
  double *u_new = malloc(node->m * svd_cutoff_idx * sizeof(double));
  if (u_new == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<node->m; j++) {
      u_new[j + i * node->m] = node->u[j + i * node->m] * s[i];
    }
  }

  // Copy new V
  double *v_new = malloc(svd_cutoff_idx * node->n * sizeof(double));
  if (v_new == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<node->n; j++) {
      v_new[j + i * node->n] = node->v[i + j * m_smaller];
    }
  }

  free(node->u); node->u = u_new;
  free(node->v); node->v = v_new;
  node->s = svd_cutoff_idx;

  free(s);

  return result;
}


/**
 * Sums up the ranks of all lower-level nodes that will be used to compute an
 * output node.
 *
 * Given a ``parent`` node to start at and its index within its level 
 * (``parent_position``), loops up the tree (through its parents) and adds up 
 * the ranks of the relevant off-diagonal leaf nodes (determined based on 
 * ``parent_position``).
 *
 * Parameters
 * ----------
 * parent
 *     Pointer to an internal node from which to start computing. Must be the
 *     parent of the internal node whose leaf nodes are being used for the
 *     multiplication.
 */
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


/**
 * Computes the matrix-matrix multiplication of non-base off-diagonal leaf 
 * nodes for any off-diagonal leaf node.
 *
 * Loops up the trees, starting with the parent nodes ``parent_left`` and 
 * ``parent_right`` and computes the appropriate matrix-matrix multiplications
 * for the pair of output nodes ``out_tr`` and ``out_bl``. The results are
 * written into their U and V arrays.
 *
 * Parameters
 * ----------
 * height
 *     The reversed height of the parent trees, i.e. the number of parent 
 *     nodes until ``parent_left`` and ``parent_right`` become ``NULL``. This
 *     is equal to the level index of the parents' children. E.g., if 
 *     ``parent_left == NULL``, ``height`` should be ``0``. If ``parent_left``
 *     and ``parent_right`` are the root nodes, ``height`` should be ``1``.
 * origin_idx
 *     The index of the parent ``out_tr`` and ``out_bl`` within the level. 
 *     E.g., the top-left-most node has index of ``0``.
 * parent_left
 *     Pointer to an internal node from the left-side matrix whose 
 *     grandchildren are being multiplied. May be ``NULL`` **if** 
 *     ``height==0``, otherwise must be fully constructed and contain data.
 *     Must have ``height-1`` non-NULL parents.
 * parent_right
 *     Pointer to an internal node from the right-side matrix whose 
 *     grandchildren are being multiplied. May be ``NULL`` **if** 
 *     ``height==0``, otherwise must be fully constructed and contain data.
 *     Must have ``height-1`` non-NULL parents.
 * out_tr
 *     Pointer to an off-diagonal leaf node to which to save the top-right 
 *     results (i.e. ``parent1->children[0] @ parent2->children[1]`` &&
 *     ``parent1->children[1] @ parent2->children[3]``). Must be fully 
 *     constructed but should not contain any data because it might be lost.
 * out_bl
 *     Pointer to an off-diagonal leaf node to which to save the top-right
 *     results (i.e. ``parent1->children[2] @ parent2->children[0]`` && 
 *     ``parent1->children[3] @ parent2->children[2]``). Must be fully
 *     constructed but should not contain any data because it might be lost
 * offsets
 *     Pointer representing an array of offsets into the U and V arrays of 
 *     higher-level nodes. Must not be ``NULL``. Must be of at least 
 *     length ``height``. Its state must be consistent with ``origin_idx``, 
 *     which can be achieved by calling this function in order and not 
 *     altering the ``offsets`` in between.
 * workspace
 *     Pointer representing an array used as a workspace matrix. Must not be 
 *     ``NULL``. Must be of size at least ``s1`` x ``s2``, where ``s1`` and 
 *     ``s2`` are the largest pair of ranks from the relevant children on the 
 *     upward path of ``parent_left`` and ``parent_right`` respectively.*
 * offset_utr_vbl_out
 *     Pointer to a single integer, into which the final offset into the 
 *     ``out_tr->u`` and ``out_bl->v`` arrays will be written. This is the 
 *     number of elements written into these arrays by this function, so 
 *     further writes should start from ``offset_utr_vbl_out``.
 * offset_vtr_ubl_out
 *     Pointer to a single integer, into which the final offset into the
 *     ``out_tr->v`` and ``out_bl->u`` arrays will be written. This is the
 *     number of elements written into these arrays by this function, so
 *     further writes should start from ``offset_vtr_ubl_out``.
 *
 * Notes
 * -----
 * Concerning *, the largest pair is the largest ``s1 x s2`` for each pair of 
 * low-rank matrices multiplied. The "relevant" nodes depend on ``origin_idx``
 * since at each level only one of the two off-diagonal leaf nodes is used for
 * the multiplication from each tree.
 */
static inline void compute_higher_level_contributions_off_diagonal(
  const int height,
  const int origin_idx,
  int divisor,
  const struct HODLRInternalNode *restrict parent_left,
  const struct HODLRInternalNode *restrict parent_right,
  struct NodeOffDiagonal *restrict const out_tr,
  struct NodeOffDiagonal *restrict const out_bl,
  int *restrict const offsets,
  double *restrict const workspace,
  unsigned int *restrict const offset_utr_vbl_out,
  unsigned int *restrict const offset_vtr_ubl_out
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
    const int s1 = 
      parent_left->children[which_child1].leaf->data.off_diagonal.s;
    const int s2 = 
      parent_right->children[which_child2].leaf->data.off_diagonal.s;
    const int m = 
      parent_left->children[which_child1].leaf->data.off_diagonal.m;
    const int n = 
      parent_left->children[which_child1].leaf->data.off_diagonal.n;

    dgemm_("T", "N", &s1, &s2, &n, &alpha,
           parent_left->children[which_child1].leaf->data.off_diagonal.v, &n,
           parent_right->children[which_child2].leaf->data.off_diagonal.u, &n,
           &beta, workspace, &s1);

    // TOP-RIGHT OUTPUT
    // Low-rank x low-rank = V* (represents V^T, but not actually transposed)
    dgemm_("N", "T", &out_tr->n, &s1, &s2, &alpha,
           parent_right->children[which_child2].leaf->data.off_diagonal.v 
           + (offsets[oidx] + out_bl->n), &m, 
           workspace, &s1, &beta, out_tr->v + offset_vtr_ubl, &out_tr->n);
    dlacpy_("A", &out_tr->m, &s1, 
            parent_left->children[which_child1].leaf->data.off_diagonal.u
            + offsets[oidx], &m,
            out_tr->u + offset_utr_vbl, &out_tr->m);

    // BOTTOM-LEFT OUTPUT
    // Low-rank x low-rank = V* (represents V^T, but not actually transposed)
    dgemm_("N", "T", &out_bl->n, &s1, &s2, &alpha,
           parent_right->children[which_child2].leaf->data.off_diagonal.v 
           + offsets[oidx], &m, 
           workspace, &s1, &beta, out_bl->v + offset_utr_vbl, &out_bl->n);
    dlacpy_("A", &out_bl->m, &s1, 
            parent_left->children[which_child1].leaf->data.off_diagonal.u
            + offsets[oidx] + out_tr->m, &m,
            out_bl->u + offset_vtr_ubl, &out_bl->m);

    offsets[oidx] += out_tr->m + out_tr->n;
    parent_left = parent_left->parent; parent_right = parent_right->parent;
    midx++; oidx++;
    divisor *= 2; parent_position /= 2;
    offset_utr_vbl += out_tr->m * s1; offset_vtr_ubl += out_tr->n * s1;
  }

  *offset_utr_vbl_out = offset_utr_vbl;
  *offset_vtr_ubl_out = offset_vtr_ubl;
}


/**
 * Sets up an off-diagonal leaf node of the output HODLR.
 *
 * Parameters
 * ----------
 * off_diagonal_left
 *     Pointer to an off-diagonal leaf node from the left-side HODLR that will
 *     be used to compute the ``out`` leaf node. Must be fully constructed 
 *     with block sizes set. Its block sizes must be identical with 
 *     ``off_diagonal_right``.
 * off_diagonal_right
 *     Pointer to an off-diagonal leaf node from the right-side HODLR that 
 *     will be used to compute the ``out`` leaf node. Must be fully 
 *     constructed with block sizes set. Its block sizes must be identical 
 *     with ``off_diagonal_left``.
 * out
 *     Pointer to an off-diagonal leaf node from the output HODLR that will be
 *     set up. Must be fully allocated, but should be completely empty.
 * s_sum
 *     Sum of the ranks of all nodes of lower levels (higher up the tree) that
 *     will be used to compute ``out``.
 */
static inline void set_up_off_diagonal(
  const struct NodeOffDiagonal *restrict const off_diagonal_left,
  const struct NodeOffDiagonal *restrict const off_diagonal_right,
  struct NodeOffDiagonal *restrict const out,
  const int s_sum
) {
  const int s_total = s_sum + off_diagonal_left->s + off_diagonal_right->s;
  out->m = off_diagonal_left->m;
  out->n = off_diagonal_left->n;
  out->s = s_total;
  out->u = malloc(s_total * out->m * sizeof(double));
  out->v = malloc(s_total * out->n * sizeof(double));
}


/**
 * Computes the matrix-matrix multiplication of the base nodes for an 
 * off-diagonal leaf node that is not one of the innermost leaves.
 *
 * Performs the following two matrix multiplications:
 *
 * 1. ``diagonal_left`` @ ``off_diagonal_right``
 * 2. ``off_diagonal_left`` @ ``diagonal_right``
 *
 * and stores the results in ``out`` by writing to the end of its U and V 
 * arrays.
 *
 * Parameters
 * ----------
 * diagonal_left
 *     Pointer to a diagonal leaf node to multiply ``off_diagonal_right``. 
 *     Must be fully constructed and contain data. Should come from the same 
 *     parent as ``off_diagonal_left`` and must share its leading dimension 
 *     (m). Also must share its second dimension (n) with the leading 
 *     dimension (m) of ``off_diagonal_right``.
 * off_diagonal_left
 *     Pointer to an off-diagonal leaf node to multiply ``diagonal_right``.
 *     Must be fully constructed and contain data. Should come from the same
 *     parent as ``hodlr_left`` and must share its leading dimension (m). Also
 *     must share its second dimension (n) with the leading dimension (m) of
 *     ``diagonal_right``.
 * diagonal_right
 *     Pointer to a diagonal leaf node to be multiplied by ``diagonal_left``. 
 *     Must be fully constructed and contain data. Should come from the same 
 *     parent as ``off_diagonal_right`` and must share its second dimension 
 *     (n). Also must share its own leading dimension (m) with the second 
 *     dimension (n) of ``off_diagonal_left``.
 * off_diagonal_right
 *     Pointer to an off-diagonal leaf node to be multiplied by 
 *     ``diagonal_left``. Must be fully constructed and contain data. Should 
 *     come from the same parent as ``diagonal_right`` and must share its 
 *     second dimension (n). Also must share its own leading dimension (m) 
 *     with the second dimension (n) of ``diagonal_left``.
 * out
 *     Pointer to an off-diagonal leaf node to append the results of the 
 *     matrix multiplication to. Must be fully constructed and contain 
 *     :c:member:`NodeOffDiagonal.u` and :c:member:`NodeOffDiagonal.v` arrays
 *     large enough to hold the results. 
 *     ``off_diagonal_right->s + off_diagonal_left->s`` columns will be 
 *     written to the end of each, so they must be large enough to hold at 
 *     least that + ``offset_u``/``offset_v`` elements.
 * offset_u
 *     The offset into the ``out->u`` array. All data written into the array
 *     will start at this index, i.e. ``out->u[offset_u]``.
 * offset_v
 *     The offset into the ``out->v`` array. All data written into the array
 *     will start at this index, i.e. ``out->v[offset_v]``.
 *
 * Notes
 * -----
 * The two operations are performed in two steps each:
 *
 * 1. ``diagonal_left`` @ ``off_diagonal_right``
 *
 *    1. ``diagonal_left->data @ off_diagonal_right->u``
 *    2. Copy ``off_diagonal_right->v`` into ``out->v``
 * 
 * 2. ``off_diagonal_left`` @ ``diagonal_right``
 *
 *    1. Copy ``off_diagonal_left->u`` into ``out->u``
 *    2. ``diagonal_right->data.T @ off_diagonal_left->v``
 */
static inline void compute_inner_off_diagonal_lowest_level(
  const struct NodeDiagonal *const diagonal_left,
  const struct NodeOffDiagonal *const off_diagonal_left,
  const struct NodeDiagonal *const diagonal_right,
  const struct NodeOffDiagonal *const off_diagonal_right,
  struct NodeOffDiagonal *restrict const out,
  unsigned int offset_u,
  unsigned int offset_v
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


/**
 * Computes HODLR-HODLR matrix multiplication for a pair of off-diagonal leaf
 * nodes on the lowest level.
 *
 * Parameters
 * ----------
 * height
 *     The height of the HODLR matrices whose multiplication is being 
 *     computed. Must be a positive number.
 * parent_position
 *     The index of ``parent1`` and ``parent2`` within the level. I.e., the
 *     top-left-most node has index of ``0``.
 * parent1
 *     Pointer to an internal node from the left-side matrix whose children 
 *     are to be multiplied. Must be fully constructed and contain data. Its 
 *     diagonal children must be diagonal leaf nodes.
 * parent2
 *     Pointer to an internal node from the right-side matrix whose children 
 *     are to be multiplied. Must be fully constructed and contain data. Its 
 *     diagonal children must be diagonal leaf nodes.
 * out_tr
 *     Pointer to an off-diagonal leaf node to which to save the top-right 
 *     results (i.e. ``parent1->children[0] @ parent2->children[1]`` &&
 *     ``parent1->children[1] @ parent2->children[3]``). Must be fully 
 *     constructed but should not contain any data because it might be lost.
 * out_bl
 *     Pointer to an off-diagonal leaf node to which to save the top-right
 *     results (i.e. ``parent1->children[2] @ parent2->children[0]`` && 
 *     ``parent1->children[3] @ parent2->children[2]``). Must be fully
 *     constructed but should not contain any data because it might be lost.
 * svd_threshold
 *     The threshold for discarding singular values when recompressing 
 *     off-diagonal nodes - any singular values smaller than one 
 *     ``svd_threshold``-th of the first singular value will be treated as 
 *     approximately zero and therefore the corresponding column vectors of 
 *     the :math:`U` and :math:`V` matrices will be discarded.
 * offsets
 *     Pointer representing an array of offsets used by
 *     :c:func:`compute_higher_level_contributions_off_diagonal`. Must not be
 *     ``NULL``. Must be of at least length ``height``. Must be consistent 
 *     with ``current_level`` and ``parent_position``, which can be achieved
 *     by calling this function in order and not altering the ``offsets`` in
 *     between.
 * workspace
 *     Pointer representing an array used as a workspace matrix. Must not be 
 *     ``NULL``. Must be of size at least ``s1`` x ``s2``, where ``s1`` and 
 *     ``s2`` is the largest pair of ranks of low-rank matrices multiplied 
 *     together.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. An error status code from :c:enum:`ErrorCode` is written into 
 *     the pointer **if an error occurs**. Must not be ``NULL`` - doing so is 
 *     undefined.
 */
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
    parent1->parent, height-1, parent_position
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

  unsigned int offset_utr_vbl, offset_vtr_ubl;
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

  if (out_tr->s < m_smaller)
    recompress(out_tr, m_larger, m_smaller, svd_threshold, ierr);
  else 
    recompress_large_s(out_tr, m_smaller, svd_threshold, ierr);

  if (out_bl->s < m_smaller)
    recompress(out_bl, m_larger, m_smaller, svd_threshold, ierr);
  else
    recompress_large_s(out_bl, m_smaller, svd_threshold, ierr);
}


/**
 * Computes the matrix-matrix multiplication of the base nodes for an 
 * off-diagonal leaf node that is not one of the innermost leaves.
 *
 * Performs the following two matrix multiplications:
 *
 * 1. ``hodlr_left`` @ ``off_diagonal_right``
 * 2. ``off_diagonal_left`` @ ``hodlr_right``
 *
 * and stores the results in ``out`` by writing to the end of its U and V 
 * arrays.
 *
 * Parameters
 * ----------
 * height
 *     The height of the ``hodlr_left`` and ``hodlr_right`` trees. Must be a 
 *     positive number.
 * hodlr_left
 *     Pointer to an internal node consisting of ``height`` levels to multiply
 *     ``off_diagonal_right``. Must be fully constructed and contain data.
 *     Should come from the same parent as ``off_diagonal_left`` and must 
 *     share leading dimension (m). Also must share its second dimension (n)
 *     with the leading dimension (m) of ``off_diagonal_right``.
 * off_diagonal_left
 *     Pointer to an off-diagonal leaf node to multiply ``hodlr_right``.
 *     Must be fully constructed and contain data. Should come from the same
 *     parent as ``hodlr_left`` and must share its leading dimension (m). Also
 *     must share its second dimension (n) with the leading dimension (m) of
 *     ``hodlr_right``.
 * hodlr_right
 *     Pointer to an internal node consisting of ``height`` levels to be 
 *     multiplied by ``hodlr_left``. Must be fully constructed and contain 
 *     data. Should come from the same parent as ``off_diagonal_right`` and
 *     must share its second dimension (n). Also must share its leading 
 *     dimension (m) with the second dimension (n) of ``off_diagonal_left``.
 * off_diagonal_right
 *     Pointer to an off-diagonal leaf node to be multiplied by ``hodlr_left``.
 *     Must be fully constructed and contain data. Should come from the same
 *     parent as ``hodlr_right`` and must share its second dimension (n). Also
 *     must share its leading dimension (m) with the second dimension (n) of
 *     ``hodlr_left``.
 * out
 *     Pointer to an off-diagonal leaf node to append the results of the 
 *     matrix multiplication to. Must be fully constructed and contain 
 *     :c:member:`NodeOffDiagonal.u` and :c:member:`NodeOffDiagonal.v` arrays
 *     large enough to hold the results. 
 *     ``off_diagonal_right->s + off_diagonal_left->s`` columns will be 
 *     written to the end of each, so they must be large enough to hold at 
 *     least that + ``offset_u``/``offset_v`` elements.
 * offset_u
 *     The offset into the ``out->u`` array. All data written into the array
 *     will start at this index, i.e. ``out->u[offset_u]``.
 * offset_v
 *     The offset into the ``out->v`` array. All data written into the array
 *     will start at this index, i.e. ``out->v[offset_v]``.
 * workspace
 *     Pointer representing an array used as a workspace matrix for the 
 *     internal node-dense matrix multiplication. Must be of size at least
 *     ``si`` x ``so``, where ``si`` is the largest rank in an internal node
 *     tree and ``so`` is the rank of the off-diagonal node (with the minimum
 *     size being the larger of the two pairs ``hodlr_left`` x 
 *     ``off_diagonal_right`` and ``off_diagonal_left`` x ``hodlr_right``).
 * queue
 *     Pointer representing an array of pointers to internal nodes, used to
 *     loop over the internal node trees ``hodlr_left`` and ``hodlr_right``.
 *     Must not be ``NULL``. Must of length at least :math:`2^{height - 1}`.
 *
 * Notes
 * -----
 * The two operations are performed in two steps each:
 *
 * 1. ``hodlr_left`` @ ``off_diagonal_right``
 *
 *    1. ``hodlr_left->data @ off_diagonal_right->u``
 *    2. Copy ``off_diagonal_right->v`` into ``out->v``
 * 
 * 2. ``off_diagonal_left`` @ ``hodlr_right``
 *
 *    1. Copy ``off_diagonal_left->u`` into ``out->u``
 *    2. ``hodlr_right->data.T @ off_diagonal_left->v``
 */
static inline void compute_other_off_diagonal_lowest_level(
  const int height,
  const struct HODLRInternalNode *restrict const hodlr_left,
  const struct NodeOffDiagonal *restrict const off_diagonal_left,
  const struct HODLRInternalNode *restrict const hodlr_right,
  const struct NodeOffDiagonal *restrict const off_diagonal_right,
  struct NodeOffDiagonal *restrict const out,
  unsigned int offset_u,
  unsigned int offset_v,
  double *restrict const workspace,
  struct HODLRInternalNode *restrict *restrict queue
) {
  // HODLR x U = U* at index=0
  multiply_internal_node_dense(
    hodlr_left, height, 
    off_diagonal_right->u, off_diagonal_right->s, off_diagonal_right->m, 
    queue, workspace, 
    out->u + offset_u, off_diagonal_right->m
  );
  // Copy V
  memcpy(out->v + offset_v, off_diagonal_right->v, 
         off_diagonal_right->n * off_diagonal_right->s * sizeof(double));

  offset_u += hodlr_left->m * off_diagonal_right->s;
  offset_v += off_diagonal_right->s * off_diagonal_right->n;

  // Copy U
  memcpy(out->u + offset_u, off_diagonal_left->u, 
         off_diagonal_left->m * off_diagonal_left->s * sizeof(double));
  // HODLR^T x V = V* at index=1 (represents V^T* but not actually transposed)
  multiply_internal_node_transpose_dense(
    hodlr_right, height, 
    off_diagonal_left->v, off_diagonal_left->s, off_diagonal_left->n, 
    queue, workspace, 
    out->v + offset_v, off_diagonal_left->n
  );
}


// TODO: Add an image to aid explanation
/**
 * Computes the result of a HODLR-HODLR matrix multiplication for a pair of 
 * off-diagonal leaf nodes that are not one of the lowest-level ones.
 *
 * Given two parent internal nodes (one from the left-side HODLR and one from
 * the right-side HODLR), computes the matrix multiply in the form of an 
 * off-diagonal leaf node.
 *
 * Parameters
 * ----------
 * height
 *     The height of the HODLR matrices whose multiplication is being 
 *     computed. Must be a positive number.
 * current_level
 *     The level index of ``parent1`` and ``parent2``. Must be a positive 
 *     number smaller than ``height``. E.g., the root node has a level of 
 *     ``0``.
 * parent_position
 *     The index of ``parent1`` and ``parent2`` within the level. I.e., the
 *     top-left-most node has index of ``0``.
 * parent1
 *     Pointer to an internal node from the left-side matrix whose children 
 *     are to be multiplied. Must be fully constructed and contain data. Its 
 *     diagonal children must be internal nodes of height 
 *     ``height - current_level - 1``.
 * parent2
 *     Pointer to an internal node from the right-side matrix whose children 
 *     are to be multiplied. Must be fully constructed and contain data. Its 
 *     diagonal children must be internal nodes of height 
 *     ``height - current_level - 1``.
 * out_tr
 *     Pointer to an off-diagonal leaf node to which to save the top-right 
 *     results (i.e. ``parent1->children[0] @ parent2->children[1]`` &&
 *     ``parent1->children[1] @ parent2->children[3]``). Must be fully 
 *     constructed but should not contain any data because it might be lost.
 * out_bl
 *     Pointer to an off-diagonal leaf node to which to save the top-right
 *     results (i.e. ``parent1->children[2] @ parent2->children[0]`` && 
 *     ``parent1->children[3] @ parent2->children[2]``). Must be fully
 *     constructed but should not contain any data because it might be lost.
 * queue
 *     Pointer representing an array of pointers to internal nodes. Used to 
 *     loop over the internal node children of ``parent1`` and ``parent2``.
 *     Must not be ``NULL``. Must be of a length of at least 
 *     :math:`2^{height - current_level - 2}`. May be empty.
 * svd_threshold
 *     The threshold for discarding singular values when recompressing 
 *     off-diagonal nodes - any singular values smaller than one 
 *     ``svd_threshold``-th of the first singular value will be treated as 
 *     approximately zero and therefore the corresponding column vectors of 
 *     the :math:`U` and :math:`V` matrices will be discarded.
 * offsets
 *     Pointer representing an array of offsets used by
 *     :c:func:`compute_higher_level_contributions_off_diagonal`. Must not be
 *     ``NULL``. Must be of at least length ``height``. Must be consistent 
 *     with ``current_level`` and ``parent_position``, which can be achieved
 *     by calling this function in order and not altering the ``offsets`` in
 *     between.
 * workspace
 *     Pointer representing an array used as a workspace matrix. Must not be 
 *     ``NULL``. Must be of size at least ``s1`` x ``s2``, where ``s1`` and 
 *     ``s2`` is the largest pair of ranks of low-rank matrices multiplied 
 *     together.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. An error status code from :c:enum:`ErrorCode` is written into 
 *     the pointer **if an error occurs**. Must not be ``NULL`` - doing so is 
 *     undefined.
 */
static inline void compute_other_off_diagonal(
  const int height,
  const int current_level,
  const int parent_position,
  const struct HODLRInternalNode *restrict const parent1,
  const struct HODLRInternalNode *restrict const parent2,
  struct NodeOffDiagonal *restrict out_tr,
  struct NodeOffDiagonal *restrict out_bl,
  struct HODLRInternalNode *restrict *restrict queue,
  const double svd_threshold,
  int *restrict offsets,
  double *restrict workspace,
  int *restrict const ierr
) {
  const int s_sum = compute_workspace_size_s_component(
    parent1->parent, current_level, parent_position
  );

  // TODO: Consider setting m and n in set_up_off_diagonal
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

  unsigned int offset_utr_vbl, offset_vtr_ubl;
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
    out_bl, offset_vtr_ubl, offset_utr_vbl, workspace, queue
  );

  int m_larger, m_smaller;
  if (out_tr->m > out_tr->n) {
    m_larger = out_tr->m; m_smaller = out_tr->n;
  } else {
    m_larger = out_tr->n; m_smaller = out_tr->m;
  }

  if (out_tr->s < m_smaller)
    recompress(out_tr, m_larger, m_smaller, svd_threshold, ierr);
  else 
    recompress_large_s(out_tr, m_smaller, svd_threshold, ierr);

  if (out_bl->s < m_smaller)
    recompress(out_bl, m_larger, m_smaller, svd_threshold, ierr);
  else
    recompress_large_s(out_bl, m_smaller, svd_threshold, ierr);
}


/**
 * Multiplies a subsection of two off-diagonal leaf nodes and adds the result 
 * to a dense matrix.
 * 
 * Given an low-rank matrix represented by :math:`U_1` and :math:`V_1` and 
 * another low-rank matrix represented by :math:`U_2` and :math:`V_2`, 
 * computes a subset of :math:`U_1 [(V_1^T U_2)V_2^T]` of size ``m`` x ``m``
 * and adds the result to the output.
 *
 * Parameters
 * ----------
 * leaf1
 *     Pointer to an off-diagonal leaf node holding a low-rank matrix which 
 *     will multiply ``leaf2``. Must be fully constructed contain data - 
 *     otherwise is undefined. The following must also be true: 
 *     ``leaf1->n == leaf2->m`` - these not matching will result in ``dgemm``
 *     failure.
 * leaf2
 *     Pointer to an off-diagonal leaf node holding a low-rank matrix which 
 *     will be multiplied by ``leaf1``. Must be fully constructed contain data
 *     - otherwise is undefined. The following must also be true: 
 *     ``leaf1->n == leaf2->m`` - these not matching will result in ``dgemm``
 *     failure.
 * offset
 *     An offset into the :math:`U_1` and :math:`V_2` arrays, used to select
 *     ``m`` rows of each matrix starting from ``offset``. Must be positive
 *     and smaller than both ``leaf1->m - m`` and ``leaf2->n - m``.
 * workspace1
 *     Pointer representing an array used as a workspace matrix for storing 
 *     the result of the first multiplication :math:`W_1 = V_1^T U_2`. Must 
 *     not be ``NULL`` and must be of at least size ``leaf1->s`` x 
 *     ``leaf2->s``. Must not overlap with any other array.
 * workspace2
 *     Pointer representing an array used as a workspace matrix for storing
 *     the result of the second multiplication :math:`W_2 = W_1 V_2^T`. Must
 *     not be ``NULL`` and must be of at least size ``leaf1->s`` x ``m``. 
 *     Must not overlap with any other array.
 * out
 *     Pointer representing an array used as the output matrix. Must not be 
 *     ``NULL`` and must be of size ``m`` x ``m``. Must not overlap with any
 *     other array.
 * m
 *     The size of the ``out`` matrix. Also the number of rows of ``U_1`` and
 *     ``V_2`` multiplied. Must be positive and smaller than both 
 *     ``leaf1->m - offset`` and ``leaf2->n - offset``.
 *
 * Notes
 * -----
 * The multi-step multiplication is performed in the following way:
 *
 * 1. :math:`W_1 = V_1^T[:, :] @ U_2[:, :]`
 * 2. :math:`W_2 = W_1 @ V_2^T[:, offset:offset+m]`
 * 3. :math:`O = O + U_1[offset:offset+m, :] @ W_2`
 *
 * where :math:`W_1`, :math:`W_2`, and :math:`O` are the ``workspace1``, 
 * ``workspace2``, and ``out`` matrices. :math:`@` signifies a matrix-matrix
 * multiplication and ``[]`` brackets and contents are used as a 2D array 
 * indexing notation in the ``numpy`` style.
 */
static inline void add_off_diagonal_contribution(
  const struct NodeOffDiagonal *restrict const leaf1,
  const struct NodeOffDiagonal *restrict const leaf2,
  const int offset,
  double *restrict workspace1,
  double *restrict workspace2,
  double *restrict out,
  const int m
) {
  const double alpha = 1.0, beta = 0.0;

  dgemm_("T", "N", &leaf1->s, &leaf2->s, &leaf1->n, &alpha,
         leaf1->v, &leaf1->n, leaf2->u, &leaf2->m,
         &beta, workspace1, &leaf1->s);

  dgemm_("N", "T", &leaf1->s, &m, &leaf2->s, &alpha,
         workspace1, &leaf1->s, leaf2->v + offset, &leaf2->n,
         &beta, workspace2, &leaf1->s);

  dgemm_("N", "N", &m, &m, &leaf1->s, &alpha,
         leaf1->u + offset, &leaf1->m, workspace2, &leaf1->s, &alpha, 
         out, &m);
}


/**
 * Computes the result of HODLR-HODLR matrix multiplication for the diagonal 
 * leaf nodes.
 *
 * Parameters
 * ----------
 *
 * hodlr1
 *     Pointer to a HODLR matrix that multiplies ``hodlr2``. Must be fully 
 *     constructed and contain data.
 * hodlr2
 *     Pointer to a HODLR matrix that is multiplied by ``hodlr1``. Must be 
 *     fully constructed and contain data.
 * out
 *     Pointer to a HODLR matrix to which to save the diagonal blocks. Must be
 *     fully allocated but the diagonal leaf nodes must be empty. Must point 
 *     to a different location from ``hodlr1`` and ``hodlr2``.
 * offsets
 *     Pointer that represents an array of offsets. Must be of the same length
 *     as the height of all the HODLR matrices.
 * workspace1
 *     Pointer representing an array used as a workspace matrix in the 
 *     low-rank expansion :c:func:`add_off_diagonal_contribution`. Must not be
 *     ``NULL`` and must be of at least size ``s1`` x ``s2``, where ``s1`` and
 *     ``s2`` are the largest rank of any off-diagonal leaf node on ``hodlr1``
 *     and ``hodlr2`` respectively.*
 * workspace2
 *     Pointer representing an array used as a workspace matrix in the 
 *     low-rank expansion :c:func:`add_off_diagonal_contribution`. Must not be 
 *     ``NULL`` and must be of at least size ``s1`` x ``n``, where ``s1`` is
 *     the largest rank of any off-diagonal leaf node, and ``n`` is the number
 *     of rows of the largest diagonal leaf node.**
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. An error status code from :c:enum:`ErrorCode` is written into 
 *     the pointer **if an error occurs**. Must not be ``NULL`` - doing so is 
 *     undefined.
 *
 * Notes
 * -----
 * Concerning * and **, technically ``s1`` and ``s2`` do not have to be the
 * largest numbers overall, rather their combination (``s1×s2``/``s1×n``) 
 * must be the largest, but the former is simpler to compute and describe. 
 * What this means in practice is that to get the lowest sizes, all the 
 * possible multiplications that need to be performed have to be checked, and
 * their ``s1×s2`` and ``s1×n`` have to be computed, with the largest values
 * used.
 */
static void compute_diagonal(
  const struct TreeHODLR *restrict const hodlr1,
  const struct TreeHODLR *restrict const hodlr2,
  struct TreeHODLR *restrict out,
  int *restrict offsets,
  double *restrict workspace1,
  double *restrict workspace2,
  int *restrict ierr
) {
  // TODO: Consider using offsets on stack
  struct HODLRInternalNode *parent_node1 = NULL, *parent_node2 = NULL;
  int idx = 0, oidx = 0;
  const double alpha = 1.0, beta = 0.0;
  int which_child1 = 0, which_child2 = 0;

  for (int parent = 0; parent < out->len_work_queue; parent++) {
    for (int _diagonal = 0; _diagonal < 2; _diagonal++) {
      const int m = hodlr1->innermost_leaves[idx]->data.diagonal.m;

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
          offsets[oidx], workspace1, workspace2, 
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


/**
 * Computes the workspace sizes for :c:func:`multiply_hodlr_hodlr`.
 *
 * Parameters
 * ----------
 * hodlr1
 *     Pointer to the HODLR matrix that will be used as the ``hodlr1`` 
 *     argument to the multiply function. Must be fully constructed with all
 *     the block sizes (including ranks) set - anything else is undefined.
 * hodlr2
 *     Pointer to the HODLR matrix that will be used as the ``hodlr2`` 
 *     argument to the multiply function. Must be fully constructed with all
 *     the block sizes (including ranks) set - anything else is undefined.
 * size1
 *     Pointer to an ``int`` into which the size of the first workspace 
 *     (``workspace1``) will be written. Must not be ``NULL``.
 * size2
 *     Pointer to an ``int`` into which the size of the second workspace
 *     (``workspace2``) will be written. Must not be ``NULL``.
 */
void compute_workspace_multiply_hodlr_hodlr(
  const struct TreeHODLR *const hodlr1,
  const struct TreeHODLR *const hodlr2,
  unsigned int *restrict const size1,
  unsigned int *restrict const size2
) {
  const int s1 = get_highest_s(hodlr1);
  const int s2 = get_highest_s(hodlr2);

  *size1 = s1 * s2;

  int largest_m = 0;
  for (int node = 0; node < 2 * hodlr1->len_work_queue; node++) {
    const int m = hodlr1->innermost_leaves[node]->data.diagonal.m;
    if (m > largest_m) largest_m = m;
  }

  *size2 = s1 * largest_m;
}


/**
 * Multiplies two HODLR matrices
 *
 * Parameters
 * ----------
 * hodlr1
 *     Pointer to a HODLR matrix. Must be fully constructed and contain data
 *     - anything else is undefined.
 * hodlr2
 *     Pointer to a HODLR matrix. Must be fully constructed and contain data -
 *     anything else is undefined. Must have the same height as ``hodlr1``.
 * out
 *     Pointer to a HODLR matrix to which to save the result. Must be 
 *     fully allocated - otherwise is undefined. Must not contain any data - 
 *     all arrays on the tree will be substituted and the data will be lost,
 *     potentially leading to memory leaks. Must have the same height as 
 *     ``hodlr1``.
 * svd_threshold
 *     The threshold for discarding singular values when recompressing 
 *     off-diagonal nodes - any singular values smaller than one 
 *     ``svd_threshold``-th of the first singular value will be treated as 
 *     approximately zero and therefore the corresponding column vectors of 
 *     the :math:`U` and :math:`V` matrices will be discarded.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. A status code from :c:enum:`ErrorCode` is written into 
 *     the pointer. Must not be ``NULL`` - doing is is undefined.
 */
void multiply_hodlr_hodlr(
  const struct TreeHODLR *restrict const hodlr1,
  const struct TreeHODLR *restrict const hodlr2,
  struct TreeHODLR *restrict out,
  const double svd_threshold,
  int *restrict ierr
) {
  unsigned int size1, size2;
  compute_workspace_multiply_hodlr_hodlr(hodlr1, hodlr2, &size1, &size2);

  double *workspace = malloc((size1 + size2) * sizeof(double)); 
  double *workspace2 = workspace + size1;
  int *offsets = calloc(hodlr1->height, sizeof(int));
  if (offsets == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return;
  }

  struct HODLRInternalNode **queue = out->work_queue;
  struct HODLRInternalNode **q1 = hodlr1->work_queue;
  struct HODLRInternalNode **q2 = hodlr2->work_queue;
  struct HODLRInternalNode **extra_queue =
    malloc(out->len_work_queue * sizeof(struct HODLRInternalNode *));

  compute_diagonal(
    hodlr1, hodlr2, out, offsets, workspace, workspace2, ierr
  );

  long n_parent_nodes = out->len_work_queue;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    queue[parent] = out->innermost_leaves[2 * parent]->parent;
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
        extra_queue, svd_threshold, offsets, workspace, ierr
      );

      queue[parent / 2] = queue[parent]->parent;
      q1[parent / 2] = q1[parent]->parent;
      q2[parent / 2] = q2[parent]->parent;
    }
  }
  
  free(offsets); free(extra_queue); free(workspace);
}

