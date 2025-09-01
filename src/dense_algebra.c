#include <stdlib.h>
#include <math.h>

#include "../include/lapack_wrapper.h"
#include "../include/tree.h"
#include "../include/error.h"
#include "../include/utils.h"
#include "../include/blas_wrapper.h"


/**
 * Computes the workspace size required for the HODLR-dense multiplication
 * functions.
 *
 * Given a HODLR tree and the non-shared dimension of the dense matrix, 
 * computes the minimum length of the workspace array used by the 
 * :c:func:`multiply_hodlr_dense` and :c:func:`multiply_dense_hodlr` functions
 * when running with that HODLR tree.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to the HODLR for which to compute the workspace size. Must not 
 *     be ``NULL`` and must be a fully constructed tree occupied by data - 
 *     anything else will result in undefined behaviour.
 * matrix_a
 *     The non-shared dimension of the dense matrix, i.e. the number of 
 *     columns for :c:func:`multiply_hodlr_dense` or the number of rows for 
 *     :c:func:`multiply_dense_hodlr`, for which to compute the workspace 
 *     sizes. Must be greater than 0, other values are undefined behaviour.
 *
 * Returns
 * -------
 * int
 *     The number of elements that the workspace array will require for the
 *     multiplication of the two matrices.
 */
int compute_multiply_hodlr_dense_workspace(
  const struct TreeHODLR *restrict const hodlr,
  const int matrix_a
) {
  return get_highest_s(hodlr) * matrix_a;
}


/**
 * Multiplies a low-rank matrix and a dense matrix.
 *
 * Given an off-diagonal leaf node (which represents a low-rank matrix) and a 
 * dense matrix, computes the product of the two as a dense matrix.
 *
 * Parameters
 * ----------
 * node
 *     Pointer to the off-diagonal node to multiply. Must not be ``NULL``. 
 *     Must point to a valid node with data allocated and set.
 * matrix
 *     Pointer representing an array containing the dense matrix to be 
 *     multiplied. Must not be ``NULL``. Must not overlap with ``workspace``
 *     or ``out``.
 * matrix_n
 *     The number of columns of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``.
 * beta2
 *     The value of ``beta`` to use for the final ``dgesdd``. Determines 
 *     whether results overwrite ``out`` or are added.
 * workspace
 *     Pointer representing a workspace array available for storing 
 *     intermediate results. Must be of size at least ``node->s`` x 
 *     ``matrix_n``. Must not overlap with ``matrix`` or ``out``. Must not be
 *     ``NULL``.
 * out
 *     Pointer representing an array to which to store the result. Must be of
 *     size at least ``m`` x ``matrix_n`` matrix, where ``m`` is the number of 
 *     rows of ``node``. Must not overlap with ``matrix`` or ``workspace``. 
 *     Must not be ``NULL``.
 * out_ld
 *     The leading dimension of ``out``.
 */
static inline void multiply_low_rank_dense(
  const struct NodeOffDiagonal *restrict const node,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  const double beta2,
  double *restrict const workspace,
  double *restrict const out,
  const int out_ld
) {
  const double alpha = 1.0, beta1 = 0.0;
  dgemm_("T", "N", &node->s, &matrix_n, &node->n, &alpha, node->v, &node->n, 
         matrix, &matrix_ld, &beta1, workspace, &node->s);

  dgemm_("N", "N", &node->m, &matrix_n, &node->s, &alpha, node->u, &node->m, 
         workspace, &node->s, &beta2, out, &out_ld);
}


/**
 * Multiplies the transpose of a low-rank matrix and a dense matrix.
 *
 * Given an off-diagonal leaf node (which represents a low-rank matrix) and a 
 * dense matrix, transposes the low-rank matrix and computes the product of 
 * the two as a dense matrix.
 *
 * Parameters
 * ----------
 * node
 *     Pointer to the off-diagonal node to multiply transposed. Must not be 
 *     ``NULL``. Must point to a valid node with data allocated and set.
 * matrix
 *     Pointer representing an array containing the dense matrix to be 
 *     multiplied. Must not be ``NULL``. Must not overlap with ``workspace``
 *     or ``out``.
 * matrix_n
 *     The number of columns of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``.
 * beta2
 *     The value of ``beta`` to use for the final ``dgesdd``. Determines 
 *     whether results overwrite ``out`` or are added.
 * workspace
 *     Pointer representing a workspace array available for storing 
 *     intermediate results. Must be of size at least ``node->s`` x 
 *     ``matrix_n``. Must not overlap with ``matrix`` or ``out``. Must not be
 *     ``NULL``.
 * out
 *     Pointer representing an array to which to store the result. Must be of
 *     size at least ``m`` x ``matrix_n`` matrix, where ``m`` is the number of 
 *     rows of ``node``. Must not overlap with ``matrix`` or ``workspace``. 
 *     Must not be ``NULL``.
 * out_ld
 *     The leading dimension of ``out``.
 * node
 *     Pointer to the off-diagonal node to multiply. It must not be
 *              ``NULL`` and must point to a valid node with correctly 
 *              allocated and set values, anything else is undefined.
 */
static inline void multiply_low_rank_transpose_dense(
  const struct NodeOffDiagonal *restrict const node,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  const double beta2,
  double *restrict const workspace,
  double *restrict const out,
  const int out_ld
) {
  const double alpha = 1.0, beta1 = 0.0;
  dgemm_("T", "N", &node->s, &matrix_n, &node->m, &alpha, node->u, &node->m, 
         matrix, &matrix_ld, &beta1, workspace, &node->s);

  dgemm_("N", "N", &node->n, &matrix_n, &node->s, &alpha, node->v, &node->n, 
         workspace, &node->s, &beta2, out, &out_ld);
}


/**
 * Multiplies two off-diagonal blocks with a dense matrix.
 *
 * Computes the matrix-matrix multiplication of the two off-diagonal leaf
 * children of an internal node with a dense matrix, and adds the results to 
 * the ``out`` matrix.
 *
 * Parameters
 * ----------
 * parent
 *     Pointer to the internal node whose off-diagonal children to multiply.
 * matrix
 *     Pointer representing an array holding the portion of the matrix to be
 *     multiplied. A ``parent->m`` x ``matrix_n`` block of rows, starting with 
 *     ``matrix[0]`` will be multiplied by ``parent``. Must not be ``NULL``.
 *     Must not overlap with ``out`` or ``workspace``.
 * matrix_n
 *     The number of columns of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``.
 * out
 *     Pointer representing an array to which the results are to be saved. A
 *     ``parent->m`` x ``matrix_n`` block of columns, starting with 
 *     ``out[0]``, will be added to ``out``
 *     This array *must* be populated - the values not being set is an 
 *     undefined behaviour. Similarly, it must not be ``NULL`` and must not
 *     overlap with ``matrix`` or ``workspace``.
 * out_ld
 *     Leading dimension of ``out``.
 * workspace
 *     Pointer representing an array that can be used as a workspace. Must be 
 *     of size at least S x ``matrix_n`` where S the greater rank between the
 *     two off-diagonal children of ``parent. Must not be ``NULL`` and must 
 *     not overlap with ``matrix`` or ``out``.
 *
 * Returns
 * -------
 * int
 *     The offset increment, i.e. the size of the block written by this 
 *     function - the next call of this function should pass in 
 *     ``matrix + returned_val``.
 */
static inline int multiply_off_diagonal_dense(
  const struct HODLRInternalNode *restrict const parent,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  double *restrict const out,
  const int out_ld,
  double *restrict const workspace
) {
  const double one = 1.0;
  const int m = parent->children[1].leaf->data.off_diagonal.m;

  multiply_low_rank_dense(
    &parent->children[1].leaf->data.off_diagonal, matrix + m, matrix_n,
    matrix_ld, one, workspace, out, out_ld
  );

  multiply_low_rank_dense(
    &parent->children[2].leaf->data.off_diagonal, matrix, matrix_n,
    matrix_ld, one, workspace, out + m, out_ld
  );

  return parent->m;
}


/**
 * Multiplies the transpose of two off-diagonal blocks with a dense matrix.
 *
 * Computes the matrix-matrix multiplication of the transpose of an internal
 * node (and its two off-diagonal children) with a dense matrix, and adds the 
 * results to the ``out`` matrix.
 *
 * Parameters
 * ----------
 * parent
 *     Pointer to the internal node whose off-diagonal children to multiply
 *     transposed.
 * matrix
 *     Pointer representing an array holding the portion of the matrix to be
 *     multiplied. A ``parent->m`` x ``matrix_n`` block of rows, starting with 
 *     ``matrix[0]`` will be multiplied by ``parent``. Must not be ``NULL``.
 *     Must not overlap with ``out`` or ``workspace``.
 * matrix_n
 *     The number of columns of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``.
 * out
 *     Pointer representing an array to which the results are to be saved. A
 *     ``parent->m`` x ``matrix_n`` block of columns, starting with 
 *     ``out[0]``, will be added to ``out``
 *     This array *must* be populated - the values not being set is an 
 *     undefined behaviour. Similarly, it must not be ``NULL`` and must not
 *     overlap with ``matrix`` or ``workspace``.
 * out_ld
 *     Leading dimension of ``out``.
 * workspace
 *     Pointer representing an array that can be used as a workspace. Must be 
 *     of size at least S x ``matrix_n`` where S the greater rank between the
 *     two off-diagonal children of ``parent. Must not be ``NULL`` and must 
 *     not overlap with ``matrix`` or ``out``.
 *
 * Returns
 * -------
 * int
 *     The offset increment, i.e. the size of the block written by this 
 *     function - the next call of this function should pass in 
 *     ``matrix + returned_val``.
 */
static inline int multiply_off_diagonal_transpose_dense(
  const struct HODLRInternalNode *restrict const parent,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  double *restrict const out,
  const int out_ld,
  double *restrict const workspace
) {
  const double one = 1.0;
  
  const int m = parent->children[1].leaf->data.off_diagonal.m;

  multiply_low_rank_transpose_dense(
    &parent->children[2].leaf->data.off_diagonal, matrix + m, matrix_n,
    matrix_ld, one, workspace, out, out_ld
  );

  multiply_low_rank_transpose_dense(
    &parent->children[1].leaf->data.off_diagonal, matrix, matrix_n,
    matrix_ld, one, workspace, out + m, out_ld
  );

  return parent->m;
}


/**
 * Performs a matrix-matrix multiplication of a HODLR and a dense matrix, 
 * returning a dense matrix.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to the HODLR tree to multiply. This must be a fully constructed 
 *     HODLR tree, filled with data. If ``NULL``, the function will 
 *     immediately abort.
 * matrix
 *     Pointer representing an array storing the dense matrix to be 
 *     multiplied. Must be of size ``matrix_ld`` x ``matrix_n`` of which a M x
 *     ``matrix_n`` submatrix will be used for the multiplication (where M is 
 *     the number of rows of ``hodlr``). Must be stored in column-major order.
 *     Must not overlap with ``out`` and must be occupied with values - either 
 *     will lead to undefined behaviour.
 *     If ``NULL``, the function will immediately abort.
 * matrix_n
 *     The number of columns of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``, i.e. the number of rows of the 
 *     full matrix. Must be greater than or equal to the number of rows of 
 *     ``hodlr``.
 * out
 *     Pointer representing an array to be used for storing the results of the
 *     multiplication. Must be of size ``out_ld`` x ``matrix_n``. Will be 
 *     stored in column-major order.
 *     Must not overlap with ``matrix`` as it leads to undefined behaviour, 
 *     but may be either filled with values (which will be overwritten) or 
 *     empty (i.e. just allocated).
 *     If ``NULL``, a new array is allocated.
 * out_ld
 *     The leading dimension of ``out``, i.e. the number of rows of the full 
 *     array. Must be greater than or equal to the number of rows of 
 *     ``hodlr``, even if ``out == NULL``, in which case ``out`` will be 
 *     allocated with size ``out_ld`` x ``matrix_n``.
 * ierr
 *     Pointer to an integer hodling the :c:member:`ErrorCode.SUCCESS` value. 
 *     Used to signal the success or failure of this function. An error status 
 *     code from :c:enum:`ErrorCode` is written into the pointer 
 *     **if an error occurs**. Must not be ``NULL`` - doing so is undefined.
 *
 * Returns
 * -------
 * double*
 *     The ``out`` array with the results of the matrix-matrix multiplication 
 *     stored inside.
 *
 * Errors
 * ------
 * INPUT_ERROR
 *     If ``hodlr`` or ``matrix`` is ``NULL`` or if ``matrix`` and ``out`` 
 *     point to the same memory location.
 * ALLOCATION_FAILURE
 *     If ``out == NULL`` and its allocation fails.
 *
 * See Also
 * --------
 * multiply_hodlr_transpose_dense : Performs a transpose.
 * multiply_dense_hodlr : Other side of multiplication.
 * multiply_internal_node_dense : Uses internal node instead of HODLR struct.
 */
double * multiply_hodlr_dense(
  const struct TreeHODLR *restrict const hodlr,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  double *restrict out,
  const int out_ld,
  int *restrict const ierr
) {
  if (hodlr == NULL || matrix == NULL || matrix == out) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  if (out == NULL) {
    out = malloc(out_ld * matrix_n * sizeof(double));
    if (out == NULL) {
      *ierr = ALLOCATION_FAILURE;
      return NULL;
    }
  }
  *ierr = SUCCESS;

  int workspace_size = compute_multiply_hodlr_dense_workspace(hodlr, matrix_n);
  double *workspace = malloc(workspace_size * sizeof(double));

  long n_parent_nodes = hodlr->len_work_queue;
  struct HODLRInternalNode **queue = hodlr->work_queue;

  int offset = 0;
  const double alpha = 1.0, beta = 0.0;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;

    for (int j = 0; j < 2; j++) {
      const int m = hodlr->innermost_leaves[2 * parent + j]->data.diagonal.m;
      dgemm_("N", "N", &m, &matrix_n, &m, &alpha, 
             hodlr->innermost_leaves[2 * parent + j]->data.diagonal.data, 
             &m, matrix + offset, &matrix_ld,
             &beta, out + offset, &out_ld);
      
      offset += m;
    }
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0;

    for (int parent = 0; parent < n_parent_nodes; parent++) {
      for (int leaf = 0; leaf < 2; leaf++) {
        offset += multiply_off_diagonal_dense(
          queue[2 * parent + leaf], matrix + offset, matrix_n, matrix_ld, 
          out + offset, out_ld, workspace
        );
      }

      queue[parent] = queue[2 * parent + 1]->parent;
    }
  }

  multiply_off_diagonal_dense(
    hodlr->root, matrix, matrix_n, matrix_ld, out, out_ld, workspace
  );

  free(workspace);
        
  return out;
}


/**
 * Multiplies a HODLR matrix represented by an internal node and a dense 
 * matrix.
 *
 * Given an internal node and its height, and a dense matrix, computes their
 * product as a dense matrix.
 *
 * Parameters
 * ----------
 * internal
 *     Pointer to the internal node representing a HODLR matrix to multiply. 
 *     Must not be ``NULL`` and must be fully constructed - anything else will
 *     result in undefined behaviour.
 * height
 *     The height of the HODLR matrix represented by ``internal``. This must 
 *     correspond with the number of internal nodes starting from ``internal`` 
 *     (including) all the way to the bottom of the tree.
 * matrix
 *     Pointer representing an array storing the dense matrix to be 
 *     multiplied. Must be of size ``matrix_ld`` x ``matrix_n`` of which a M x
 *     ``matrix_n`` submatrix will be used for the multiplication (where M is 
 *     the number of rows of ``hodlr``). Must be stored in column-major order.
 *     Must not overlap with ``out`` and must be occupied with values - either 
 *     will lead to undefined behaviour.
 *     If ``NULL``, the function will immediately abort.
 * matrix_n
 *     The number of columns of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``, i.e. the number of rows of the 
 *     full matrix. Must be greater than or equal to the number of rows of 
 *     ``hodlr``.
 * queue
 *     Pointer representing an array of pointers to internal nodes. This is a 
 *     workspace array used to loop over the tree. Must be of length at least
 *     :math:`2^{height-1}`. Must not be ``NULL``.
 * workspace
 *     Pointer representing an array to be used as a workspace matrix. Must be 
 *     of size at least ``s`` x ``matrix_n`` matrix, where ``s`` is the 
 *     largest rank of any off-diagonal leaf node on the ``internal`` tree. 
 *     Must not be ``NULL``.
 * out
 *     Pointer representing an array to be used for storing the results of the
 *     multiplication. Must be of size ``out_ld`` x ``matrix_n``. Will be 
 *     stored in column-major order.
 *     Must not overlap with ``matrix`` as it leads to undefined behaviour, 
 *     but may be either filled with value (which will be overwritten) or 
 *     empty (i.e. just allocated).
 *     If ``NULL``, a new array is allocated.
 * out_ld
 *     The leading dimension of ``out``, i.e. the number of rows of the full 
 *     array. Must be greater than or equal to the number of rows of 
 *     ``hodlr``, even if ``out == NULL``, in which case ``out`` will be 
 *     allocated with size ``out_ld`` x ``matrix_n``.
 *
 * See Also
 * --------
 * multiply_hodlr_dense : Uses HODLR struct instead of internal node.
 * multiply_internal_node_transpose_dense : Performs transpose.
 * multiply_dense_internal_node : Other side of multiplication.
 */
void multiply_internal_node_dense(
  const struct HODLRInternalNode *restrict const internal,
  const int height,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  const struct HODLRInternalNode **queue,
  double *restrict workspace,
  double *restrict out,
  const int out_ld
) {
  const double alpha = 1.0, beta = 0.0;
  const int m = internal->children[1].leaf->data.off_diagonal.m;

  multiply_low_rank_dense(&internal->children[1].leaf->data.off_diagonal,
                          matrix + m, matrix_n, matrix_ld, beta,
                          workspace, out, out_ld);

  multiply_low_rank_dense(&internal->children[2].leaf->data.off_diagonal,
                          matrix, matrix_n, matrix_ld, beta,
                          workspace, out + m, out_ld);

  int idx = 0, offset = 0;
  int len_queue = 1, q_next_node_density = (int)pow(2, height-1);
  int q_current_node_density = q_next_node_density;

  queue[0] = internal;
  for (int _ = 1; _ < height; _++) {
    q_next_node_density /= 2;
    offset = 0;

    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;
      for (int child = 0; child < 4; child += 3) {
        offset += multiply_off_diagonal_dense(
          queue[idx]->children[child].internal, 
          matrix + offset, matrix_n, matrix_ld, 
          out + offset, out_ld, workspace
        );
      }

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[idx]->children[3].internal;
      queue[idx] = queue[idx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  offset = 0;
  for (int node = 0; node < len_queue; node++) {
    for (int child = 0; child < 4; child+=3) {
      const int m = queue[node]->children[child].leaf->data.diagonal.m;
      dgemm_("N", "N", &m, &matrix_n, &m, &alpha, 
              queue[node]->children[child].leaf->data.diagonal.data, 
              &m, matrix + offset, &matrix_ld,
              &alpha, out + offset, &out_ld);
      offset += m;
    }
  }
}


/**
 * Performs a matrix-matrix multiplication of the transpose of a HODLR and a 
 * dense matrix, returning a dense matrix.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to the HODLR tree to multiply transposed. This must be a fully 
 *     constructed HODLR tree, filled with data. If ``NULL``, the function 
 *     will immediately abort.
 * matrix
 *     Pointer representing an array storing the dense matrix to be 
 *     multiplied. Must be of size ``matrix_ld`` x ``matrix_n`` of which a M x
 *     ``matrix_n`` submatrix will be used for the multiplication (where M is 
 *     the number of rows of ``hodlr``). Must be stored in column-major order.
 *     Must not overlap with ``out`` and must be occupied with values - either 
 *     will lead to undefined behaviour.
 *     If ``NULL``, the function will immediately abort.
 * matrix_n
 *     The number of columns of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``, i.e. the number of rows of the 
 *     full matrix. Must be greater than or equal to the number of rows of 
 *     ``hodlr``.
 * out
 *     Pointer representing an array to be used for storing the results of the
 *     multiplication. Must be of size ``out_ld`` x ``matrix_n``. Will be 
 *     stored in column-major order.
 *     Must not overlap with ``matrix`` as it leads to undefined behaviour, 
 *     but may be either filled with value (which will be overwritten) or 
 *     empty (i.e. just allocated).
 *     If ``NULL``, a new array is allocated.
 * out_ld
 *     The leading dimension of ``out``, i.e. the number of rows of the full 
 *     array. Must be greater than or equal to the number of rows of 
 *     ``hodlr``, even if ``out == NULL``, in which case ``out`` will be 
 *     allocated with size ``out_ld`` x ``matrix_n``.
 * ierr
 *     Pointer to an integer hodling the :c:member:`ErrorCode.SUCCESS` value. 
 *     Used to signal the success or failure of this function. An error status 
 *     code from :c:enum:`ErrorCode` is written into the pointer 
 *     **if an error occurs**. Must not be ``NULL`` - doing so is undefined.
 *
 * Returns
 * -------
 * double*
 *     The ``out`` array with the results of the matrix-matrix multiplication 
 *     stored inside.
 *
 * Errors
 * ------
 * INPUT_ERROR
 *     If ``hodlr`` or ``matrix`` is ``NULL`` or if ``matrix`` and ``out`` 
 *     point to the same memory location.
 * ALLOCATION_FAILURE
 *     If ``out == NULL`` and its allocation fails.
 *
 * See Also
 * --------
 * multiply_hodlr_dense : Does not transpose.
 * multiply_internal_node_transpose_dense : Uses internal node not HODLR struct.
 */
double * multiply_hodlr_transpose_dense(
  const struct TreeHODLR *restrict const hodlr,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  double *restrict out,
  const int out_ld,
  int *restrict const ierr
) {
  if (hodlr == NULL || matrix == NULL || matrix == out) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  if (out == NULL) {
    out = malloc(out_ld * matrix_n * sizeof(double));
    if (out == NULL) {
      *ierr = ALLOCATION_FAILURE;
      return NULL;
    }
  }
  *ierr = SUCCESS;

  int workspace_size = compute_multiply_hodlr_dense_workspace(hodlr, matrix_n);
  double *workspace = malloc(workspace_size * sizeof(double));

  multiply_internal_node_transpose_dense(
    hodlr->root, hodlr->height, matrix, matrix_n, matrix_ld, 
    hodlr->work_queue, workspace, out, out_ld
  );

  return out;
}


/**
 * Multiplies the transpose of a HODLR matrix represented by an internal node 
 * and a dense matrix.
 *
 * Given an internal node and its height, and a dense matrix, computes their
 * product as a dense matrix.
 *
 * Parameters
 * ----------
 * internal
 *     Pointer to the internal node representing a HODLR matrix to multiply
 *     transposed. Must not be ``NULL`` and must be fully constructed - 
 *     anything else will result in undefined behaviour.
 * height
 *     The height of the HODLR matrix represented by ``internal``. This must 
 *     correspond with the number of internal nodes starting from ``internal`` 
 *     (including) all the way to the bottom of the tree.
 * matrix
 *     Pointer representing an array storing the dense matrix to be 
 *     multiplied. Must be of size ``matrix_ld`` x ``matrix_n`` of which a M x
 *     ``matrix_n`` submatrix will be used for the multiplication (where M is 
 *     the number of rows of ``hodlr``). Must be stored in column-major order.
 *     Must not overlap with ``out`` and must be occupied with values - either 
 *     will lead to undefined behaviour.
 *     If ``NULL``, the function will immediately abort.
 * matrix_n
 *     The number of columns of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``, i.e. the number of rows of the 
 *     full matrix. Must be greater than or equal to the number of rows of 
 *     ``hodlr``.
 * queue
 *     Pointer representing an array of pointers to internal nodes. This is a 
 *     workspace array used to loop over the tree. Must be of length at least
 *     :math:`2^{height-1}`. Must not be ``NULL``.
 * workspace
 *     Pointer representing an array to be used as a workspace matrix. Must be 
 *     of size at least ``s`` x ``matrix_n`` matrix, where ``s`` is the 
 *     largest rank of any off-diagonal leaf node on the ``internal`` tree. 
 *     Must not be ``NULL``.
 * out
 *     Pointer representing an array to be used for storing the results of the
 *     multiplication. Must be of size ``out_ld`` x ``matrix_n``. Will be 
 *     stored in column-major order.
 *     Must not overlap with ``matrix`` as it leads to undefined behaviour, 
 *     but may be either filled with value (which will be overwritten) or 
 *     empty (i.e. just allocated).
 *     If ``NULL``, a new array is allocated.
 * out_ld
 *     The leading dimension of ``out``, i.e. the number of rows of the full 
 *     array. Must be greater than or equal to the number of rows of 
 *     ``hodlr``, even if ``out == NULL``, in which case ``out`` will be 
 *     allocated with size ``out_ld`` x ``matrix_n``.
 *
 * See Also
 * --------
 * multiply_hodlr_transpose_dense : Uses HODLR struct instead of internal node.
 * multiply_internal_node_dense : Does not transpose.
 */
void multiply_internal_node_transpose_dense(
  const struct HODLRInternalNode *restrict const internal,
  const int height,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  const struct HODLRInternalNode **queue,
  double *restrict const workspace,
  double *restrict const out,
  const int out_ld
) {
  const double alpha = 1.0, beta = 0.0;
  const int m = internal->children[1].leaf->data.off_diagonal.m;

  multiply_low_rank_transpose_dense(
    &internal->children[2].leaf->data.off_diagonal, matrix + m, matrix_n, 
    matrix_ld, beta, workspace, out, out_ld
  );

  multiply_low_rank_transpose_dense(
    &internal->children[1].leaf->data.off_diagonal, matrix, matrix_n, 
    matrix_ld, beta, workspace, out + m, out_ld
  );

  int len_queue = 1, q_next_node_density = (int)pow(2, height-1);
  int q_current_node_density = q_next_node_density;
  int idx = 0, offset = 0;

  queue[0] = internal;
  for (int _ = 1; _ < height; _++) {
    q_next_node_density /= 2;
    offset = 0;

    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;
      for (int child = 0; child < 4; child += 3) {
        offset += multiply_off_diagonal_transpose_dense(
          queue[idx]->children[child].internal,
          matrix + offset, matrix_n, matrix_ld, out + offset, out_ld, workspace
        );
      }

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[idx]->children[3].internal;
      queue[idx] = queue[idx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  offset = 0;
  for (int node = 0; node < len_queue; node++) {
    for (int child = 0; child < 4; child+=3) {
      const int m = queue[node]->children[child].leaf->data.diagonal.m;
      dgemm_("T", "N", &m, &matrix_n, &m, &alpha, 
              queue[node]->children[child].leaf->data.diagonal.data, 
              &m, matrix + offset, &matrix_ld,
              &alpha, out + offset, &out_ld);
      offset += m;
    }
  }
}


/**
 * Multiplies a dense matrix and a low-rank matrix.
 *
 * Given a dense matrix and an off-diagonal node (which represents a low-rank 
 * matrix), computies the product of the two as a dense matrix.
 *
 * Parameters
 * ----------
 * node
 *     Pointer to the off-diagonal node to be multiplied. Must not be 
 *     ``NULL``. Must point to a valid node with data allocated and set.
 * matrix
 *     Pointer representing an array containing the dense matrix to multiply.
 *     Must not be ``NULL``. Must not overlap with ``workspace`` or ``out``.
 * matrix_m
 *     The number of rows of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``.
 * beta2
 *     The value of ``beta`` to use for the final ``dgesdd``. Determines 
 *     whether results overwrite ``out`` or are added.
 * workspace
 *     Pointer representing a workspace array available for storing 
 *     intermediate results. Must be of size at least ``matrix_m`` x 
 *     ``node->s``. Must not overlap with ``matrix`` or ``out``. Must not be
 *     ``NULL``.
 * out
 *     Pointer representing an array to which to store the result. Must be of
 *     size at least ``matrix_m`` x ``m`` matrix, where ``m`` is the number of 
 *     rows of ``node``. Must not overlap with ``matrix`` or ``workspace``. 
 *     Must not be ``NULL``.
 * out_ld
 *     The leading dimension of ``out``.
 */
static inline void multiply_dense_low_rank(
  const struct NodeOffDiagonal *restrict const node,
  const double *restrict const matrix,
  const int matrix_m,
  const int matrix_ld,
  const double beta2,
  double *restrict workspace,
  double *restrict out,
  const int out_ld
) {
  const double alpha = 1.0, beta1 = 0.0;
  dgemm_("N", "N", &matrix_m, &node->s, &node->m, &alpha, matrix, &matrix_ld, 
         node->u, &node->m, &beta1, workspace, &matrix_m);

  dgemm_("N", "T", &matrix_m, &node->n, &node->s, &alpha, workspace, 
         &matrix_m, node->v, &node->n, &beta2, out, &out_ld);
}


/**
 * Multiplies a dense matrix with two off-diagonal blocks.
 *
 * Computes the matrix-matrix multiplication of a dense matrix with the two 
 * off-diagonal leaf children of an internal node, and adds the results to 
 * the ``out`` matrix.
 *
 * Parameters
 * ----------
 * parent
 *     Pointer to the internal node whose off-diagonal children are to be 
 *     multiplied.
 * matrix
 *     Pointer representing an array holding the portion of the matrix to
 *     multiply. A ``matrix_m`` x ``parent->m`` block of columns, starting 
 *     with ``matrix[0]`` will be multiply ``parent``. Must not be ``NULL``.
 *     Must not overlap with ``out`` or ``workspace``.
 * matrix_m
 *     The number of rows of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``.
 * out
 *     Pointer representing an array to which the results are to be saved. A
 *     ``matrix_m`` x ``parent->m`` block of rows, starting with ``out[0]``, 
 *     will be added to ``out``
 *     This array *must* be populated - the values not being set is an 
 *     undefined behaviour. Similarly, it must not be ``NULL`` and must not
 *     overlap with ``matrix`` or ``workspace``.
 * out_ld
 *     Leading dimension of ``out``.
 * workspace
 *     Pointer representing an array that can be used as a workspace. Must be 
 *     of size at least S x ``matrix_m`` where S the greater rank between the
 *     two off-diagonal children of ``parent. Must not be ``NULL`` and must 
 *     not overlap with ``matrix`` or ``out``.
 *
 * Returns
 * -------
 * int
 *     The offset increment, i.e. the size of the block written by this 
 *     function - the next call of this function should pass in 
 *     ``matrix + returned_val * matrix_ld``.
 */
static inline int multiply_dense_off_diagonal(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict matrix,
  const int matrix_m,
  const int matrix_ld,
  double *restrict out,
  const int out_ld,
  double *restrict workspace
) {
  const double one = 1.0;
  const int m = parent->children[1].leaf->data.off_diagonal.m;

  multiply_dense_low_rank(
    &parent->children[1].leaf->data.off_diagonal, matrix, 
    matrix_m, matrix_ld, one, workspace, out + m * out_ld, out_ld
  );

  multiply_dense_low_rank(
    &parent->children[2].leaf->data.off_diagonal, matrix + m * matrix_ld, 
    matrix_m, matrix_ld, one, workspace, out, out_ld
  );

  return parent->m;
}


/**
 * Performs a matrix-matrix multiplication of a HODLR and a dense matrix, 
 * returning a dense matrix.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to the HODLR tree to be multiplied. Must be a fully constructed 
 *     HODLR tree, filled with data. If ``NULL``, the function will 
 *     immediately abort.
 * matrix
 *     Pointer representing an array storing the dense matrix to multiply.
 *     Must be of size ``matrix_ld`` x M of which a ``matrix_m`` x M submatrix 
 *     will be used for the multiplication (where M is the number of rows of 
 *     ``hodlr``). Must be stored in column-major order.
 *     Must not overlap with ``out`` and must be occupied with values - either 
 *     will lead to undefined behaviour.
 *     If ``NULL``, the function will immediately abort.
 * matrix_m
 *     The number of rows of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``, i.e. the number of rows of the 
 *     full matrix. Must be greater than or equal to ``matrix_m``. 
 * out
 *     Pointer representing an array to be used for storing the results of the
 *     multiplication. Must be of size ``out_ld`` x M. Will be stored in 
 *     column-major order.
 *     Must not overlap with ``matrix`` as it leads to undefined behaviour, 
 *     but may be either filled with values (which will be overwritten) or 
 *     empty (i.e. just allocated).
 *     If ``NULL``, a new array is allocated.
 * out_ld
 *     The leading dimension of ``out``, i.e. the number of rows of the full 
 *     array. Must be greater than or equal to ``matrix_m``, even if 
 *     ``out == NULL``, in which case ``out`` will be allocated with size 
 *     ``out_ld`` x M.
 * ierr
 *     Pointer to an integer hodling the :c:member:`ErrorCode.SUCCESS` value. 
 *     Used to signal the success or failure of this function. An error status 
 *     code from :c:enum:`ErrorCode` is written into the pointer 
 *     **if an error occurs**. Must not be ``NULL`` - doing so is undefined.
 *
 * Returns
 * -------
 * double*
 *     The ``out`` array with the results of the matrix-matrix multiplication 
 *     stored inside.
 *
 * Errors
 * ------
 * INPUT_ERROR
 *     If ``hodlr`` or ``matrix`` is ``NULL`` or if ``matrix`` and ``out`` 
 *     point to the same memory location.
 * ALLOCATION_FAILURE
 *     If ``out == NULL`` and its allocation fails.
 *
 * See Also
 * --------
 * multiply_hodlr_dense : Other side of multiplication.
 * multiply_dense_internal_node : Uses an internal node instead of HODLR struct.
 */
double * multiply_dense_hodlr(
  const struct TreeHODLR *restrict const hodlr,
  const double *restrict const matrix,
  const int matrix_m,
  const int matrix_ld,
  double *restrict out,
  const int out_ld,
  int *restrict const ierr
) {
  if (hodlr == NULL || matrix == NULL || matrix == out) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  if (out == NULL) {
    out = malloc(out_ld * hodlr->root->m * sizeof(double));
    if (out == NULL) {
      *ierr = ALLOCATION_FAILURE;
      return NULL;
    }
  }
  *ierr = SUCCESS;

  int workspace_size = compute_multiply_hodlr_dense_workspace(hodlr, matrix_m);
  double *workspace = malloc(workspace_size * sizeof(double));

  int offset = 0;
  const double alpha = 1.0, beta = 0.0;

  long n_parent_nodes = hodlr->len_work_queue;
  struct HODLRInternalNode **queue = hodlr->work_queue;
  
  for (int parent = 0; parent < n_parent_nodes; parent++) {
    queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;

    for (int j = 0; j < 2; j++) {
      const int m = hodlr->innermost_leaves[2 * parent + j]->data.diagonal.m;
      dgemm_("N", "N", &matrix_m, &m, &m, &alpha, 
             matrix + offset * matrix_ld, &matrix_ld,
             hodlr->innermost_leaves[2 * parent + j]->data.diagonal.data, 
             &m, &beta, out + offset * out_ld, &out_ld);
      
      offset += m;
    }
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0;

    for (int parent = 0; parent < n_parent_nodes; parent++) {
      for (int leaf = 0; leaf < 2; leaf++) {
        offset += multiply_dense_off_diagonal(
          queue[2 * parent + leaf], 
          matrix + offset * matrix_ld, matrix_m, matrix_ld, 
          out + offset * out_ld, out_ld, workspace
        );
      }

      queue[parent] = queue[2 * parent + 1]->parent;
    }
  }

  multiply_dense_off_diagonal(
    hodlr->root, matrix, matrix_m, matrix_ld, out, out_ld, workspace
  );

  free(workspace);
        
  return out;
}


/**
 * Multiplies a dense matrix by a HODLR matrix represented by an internal 
 * node.
 *
 * Given a dense matrix, and an internal node and its height, computes their
 * product as a dense matrix.
 *
 * Parameters
 * ----------
 * internal
 *     Pointer to the internal node representing a HODLR matrix to be
 *     multiplied. Must not be ``NULL`` and must be fully constructed - 
 *     anything else will result in undefined behaviour.
 * height
 *     The height of the HODLR matrix represented by ``internal``. This must 
 *     correspond with the number of internal nodes starting from ``internal`` 
 *     (including) all the way to the bottom of the tree.
 * matrix
 *     Pointer representing an array storing the dense matrix to multiply.
 *     Must be of size ``matrix_ld`` x M of which a ``matrix_m`` x M submatrix 
 *     will be used for the multiplication (where M is the number of rows of 
 *     ``hodlr``). Must be stored in column-major order.
 *     Must not overlap with ``out`` and must be occupied with values - either 
 *     will lead to undefined behaviour.
 *     If ``NULL``, the function will immediately abort.
 * matrix_m
 *     The number of rows of ``matrix``.
 * matrix_ld
 *     The leading dimension of ``matrix``, i.e. the number of rows of the 
 *     full matrix. Must be greater than or equal to ``matrix_m``. 
 * queue
 *     Pointer representing an array of pointers to internal nodes. This is a 
 *     workspace array used to loop over the tree. Must be of length at least
 *     :math:`2^{height-1}`. Must not be ``NULL``.
 * workspace
 *     Pointer representing an array to be used as a workspace matrix. Must be 
 *     of size at least ``s`` x ``matrix_m`` matrix, where ``s`` is the 
 *     largest rank of any off-diagonal leaf node on the ``internal`` tree. 
 *     Must not be ``NULL``.
 * out
 *     Pointer representing an array to be used for storing the results of the
 *     multiplication. Must be of size ``out_ld`` x M. Will be stored in 
 *     column-major order.
 *     Must not overlap with ``matrix`` as it leads to undefined behaviour, 
 *     but may be either filled with values (which will be overwritten) or 
 *     empty (i.e. just allocated).
 *     If ``NULL``, a new array is allocated.
 * out_ld
 *     The leading dimension of ``out``, i.e. the number of rows of the full 
 *     array. Must be greater than or equal to ``matrix_m``, even if 
 *     ``out == NULL``, in which case ``out`` will be allocated with size 
 *     ``out_ld`` x M.
 *
 * See Also
 * --------
 * multiply_dense_hodlr : Takes a HODLR struct instead of internal node.
 * multiply_internal_node_dense : Other side of multiplication.
 */
void multiply_dense_internal_node(
  const struct HODLRInternalNode *restrict const internal,
  const int height,
  const double *restrict const matrix,
  const int matrix_m,
  const int matrix_ld,
  const struct HODLRInternalNode **queue,
  double *restrict const workspace,
  double *restrict const out,
  const int out_ld
) {
  const double alpha = 1.0, beta = 0.0;
  const int m = internal->children[1].leaf->data.off_diagonal.m;

  multiply_dense_low_rank(
    &internal->children[1].leaf->data.off_diagonal,
    matrix, matrix_m, matrix_ld, beta,
    workspace, out + m * out_ld, out_ld
  );

  multiply_dense_low_rank(
    &internal->children[2].leaf->data.off_diagonal,
    matrix + m * matrix_ld, matrix_m, matrix_ld, beta,
    workspace, out, out_ld
  );

  int len_queue = 1, q_next_node_density = (int)pow(2, height-1);
  int q_current_node_density = q_next_node_density;
  int idx = 0;

  queue[0] = internal;
  for (int _ = 1; _ < height; _++) {
    q_next_node_density /= 2;
    int offset = 0;

    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;
      for (int child = 0; child < 4; child += 3) {
        offset += multiply_dense_off_diagonal(
          queue[idx]->children[child].internal,
          matrix + offset * matrix_ld, matrix_m, matrix_ld, 
          out + offset * out_ld, out_ld, workspace
        );
      }

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[idx]->children[3].internal;
      queue[idx] = queue[idx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  int offset = 0;
  for (int node = 0; node < len_queue; node++) {
    for (int child = 0; child < 4; child+=3) {
      const int m = queue[node]->children[child].leaf->data.diagonal.m;
      dgemm_("N", "N", &matrix_m, &m, &m, &alpha, 
             matrix + offset * matrix_ld, &matrix_ld,
             queue[node]->children[child].leaf->data.diagonal.data, &m,
             &alpha, out + offset * out_ld, &out_ld);
      offset += m;
    }
  }
}

