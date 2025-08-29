#include <stdlib.h>
#include <math.h>

#include "../include/lapack_wrapper.h"
#include "../include/tree.h"
#include "../include/utils.h"
#include "../include/blas_wrapper.h"


/**
 * Computes the workspace size(s) required for the HODLR-dense multiplication
 * functions.
 *
 * Given a HODLR tree and the non-shared dimension of the dense matrix, 
 * computes the minimum lengths of the two workspace arrays used by the 
 * :c:func:`multiply_hodlr_dense` and :c:func:`multiply_dense_hodlr` functions
 * when running with that HODLR tree.
 *
 * :param hodlr: Pointer to a HODLR tree for which to compute the workspace 
 *               sizes. Must not be ``NULL`` and must be filled with data, 
 *               otherwise is undefined behaviour.
 * :param matrix_a: The non-shared dimension of the dense matrix, i.e. the 
 *                  number of columns for :c:func:`multiply_hodlr_dense` or
 *                  the number of rows for :c:func:`multiply_dense_hodlr`, for 
 *                  which to compute the workspace sizes. Must be greater than 
 *                  0, other values are undefined behaviour.
 */
int compute_multiply_hodlr_dense_workspace(
  const struct TreeHODLR *restrict const hodlr,
  const int matrix_a
) {
  int s = 0, idx = 0;
  long n_parent_nodes = hodlr->len_work_queue;

  struct HODLRInternalNode **queue = hodlr->work_queue;

  int highest_s = get_highest_s(hodlr);

  return highest_s * matrix_a;
}


/**
 * Multiplies a low-rank matrix and a dense matrix.
 *
 * Given an off-diagonal node (which represents a low-rank matrix) and a 
 * dense matrix, computes the product of the two as a dense matrix.
 *
 * :param node: A pointer to the off-diagonal node to multiply. It must not be
 *              ``NULL`` and must point to a valid node with correctly 
 *              allocated and set values, anything else is undefined.
 * :param matrix: A pointer to an array containing the dense matrix to be used
 *                for the multiplication. Must not be ``NULL``. Must not 
 *                overlap with any other pointers.
 * :param matrix_n: The number of columns of ``matrix``.
 * :param matrix_ld: The leading dimension of ``matrix``.
 * :param beta2: The value of ``beta`` to use for the final ``dgesdd``.
 *               Determines whether results overwrite ``out`` or are added.
 * :param workspace: A pointer to a workspace array to be used. Must be large
 *                   enough to accomodate a ``s`` x ``matrix_n`` matrix where
 *                   ``s`` is the number of stored singular values on 
 *                   ``node``. Must not overlap with any other pointers.
 * :param out: A pointer to an array to which to store the result. Must be 
 *             large enough to accomodate a ``m`` x ``matrix_n`` matrix, where
 *             ``m`` is the number of rows of ``node``. Must not overlap with
 *             any other pointers.
 * :param out_ld: The leading dimension of ``out``.
 */
static inline void multiply_low_rank_dense(
  const struct NodeOffDiagonal *restrict node,
  const double *restrict matrix,
  const int matrix_n,
  const int matrix_ld,
  const double beta2,
  double *workspace,
  double *out,
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
 * Given an off-diagonal node (which represents a low-rank matrix) and a 
 * dense matrix, transposes the low-rank matrix and computes the product of 
 * the two as a dense matrix.
 *
 * :param node: A pointer to the off-diagonal node to multiply. It must not be
 *              ``NULL`` and must point to a valid node with correctly 
 *              allocated and set values, anything else is undefined.
 * :param matrix: A pointer to an array containing the dense matrix to be used
 *                for the multiplication. Must not be ``NULL``. Must not 
 *                overlap with any other pointers.
 * :param matrix_n: The number of columns of ``matrix``.
 * :param matrix_ld: The leading dimension of ``matrix``.
 * :param beta2: The value of ``beta`` to use for the final ``dgesdd``. 
 *               Determines whether results overwrite ``out`` or are added.
 * :param workspace: A pointer to a workspace array to be used. Must be large
 *                   enough to accomodate a ``s`` x ``matrix_n`` matrix where
 *                   ``s`` is the number of stored singular values on 
 *                   ``node``. Must not overlap with any other pointers.
 * :param out: A pointer to an array to which to store the result. Must be 
 *             large enough to accomodate a ``m`` x ``matrix_n`` matrix, where
 *             ``m`` is the number of rows of ``node``. Must not overlap with
 *             any other pointers.
 * :param out_ld: The leading dimension of ``out``.
 */
static inline void multiply_low_rank_transpose_dense(
  const struct NodeOffDiagonal *restrict node,
  const double *restrict matrix,
  const int matrix_n,
  const int matrix_ld,
  const double beta2,
  double *workspace,
  double *out,
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
 * Computes the matrix-matrix multiplication of the two off-diagonal blocks 
 * of a tree HODLR matrix internal node with a dense matrix, and adds the 
 * results to the ``out`` matrix.
 *
 * :param parent: Pointer to an internal node holding the off-diagonal nodes
 *                to multiply.
 * :param matrix: Pointer to an array holding the matrix being multiplied.
 *                This should be a pointer to the start of the array, and the
 *                ``offset_ptr`` parameter should be used for aligning the 
 *                matrix and the HODLR blocks. Must be in column-major order.
 *                Must not overlap with any of the other pointers - doing so
 *                is an undefined behaviour.
 * :param matrix_n: The number of columns of ``matrix``.
 * :param matrix_ld: The leading dimension of ``matrix``, i.e. the number of
 *                   rows of the full array. Must be greater or equal to the
 *                   number of rows of the full HODLR matrix.
 * :param out: Pointer to an array to which the results are to be saved. This
 *             should be a pointer to the start of the array, and the 
 *             ``offset_ptr`` parameter should be used for aligning it with 
 *             the HODLR blocks.
 *             This array *must* be populated since the results are added to 
 *             it. The values not being set is an undefined behaviour.
 *             Must not overlap with any of the other pointers - doing so is 
 *             an undefined behaviour.
 * :param out_ld: Leading dimension of ``out``, i.e. the number of rows of the
 *                full array. Must be greater than or equal to the number of
 *                rows of the full HODLR matrix.
 * :param workspace: Pointer to an array that can be used as a workspace. 
 *                   Must be of at least size S x N where S is the number of 
 *                   retained singular values (the greater number between the
 *                   two off-diagonal nodes on ``parent``) and N is 
 *                   ``matrix_n``.
 *                   Must not overlap with any of the other arrays - doing 
 *                   so is an undefined behaviour.
 * :param offset: The current offset into ``matrix`` and ``out`` arrays.
 *
 * :return: The new offset.
 */
static inline int multiply_off_diagonal_dense(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict matrix,
  const int matrix_n,
  const int matrix_ld,
  double *restrict out,
  const int out_ld,
  double *restrict workspace,
  const int offset
) {
  const double one = 1.0;
  const int offset2 = offset + parent->children[1].leaf->data.off_diagonal.m;

  multiply_low_rank_dense(
    &parent->children[1].leaf->data.off_diagonal, matrix + offset2, matrix_n,
    matrix_ld, one, workspace, out + offset, out_ld
  );

  multiply_low_rank_dense(
    &parent->children[2].leaf->data.off_diagonal, matrix + offset, matrix_n,
    matrix_ld, one, workspace, out + offset2, out_ld
  );

  return offset2 + parent->children[1].leaf->data.off_diagonal.n;
}


/**
 * Multiplies the transpose of two off-diagonal blocks with a dense matrix.
 *
 * Computes the matrix-matrix multiplication of the transpose of two 
 * off-diagonal blocks of a tree HODLR matrix internal node with a dense 
 * matrix, and adds the results to the ``out`` matrix.
 *
 * The two off-diagonal blocks are fully transposed - the two low-rank 
 * matrices are switched and each is transposed. (Of course, this is not 
 * performed separately but at the same time as the multiplication.)
 *
 * :param parent: Pointer to an internal node holding the off-diagonal nodes
 *                to multiply.
 * :param matrix: Pointer to an array holding the matrix being multiplied.
 *                This should be a pointer to the start of the array, and the
 *                ``offset_ptr`` parameter should be used for aligning the 
 *                matrix and the HODLR blocks. Must be in column-major order.
 *                Must not overlap with any of the other pointers - doing so
 *                is an undefined behaviour.
 * :param matrix_n: The number of columns of ``matrix``.
 * :param matrix_ld: The leading dimension of ``matrix``, i.e. the number of
 *                   rows of the full array. Must be greater or equal to the
 *                   number of rows of the full HODLR matrix.
 * :param out: Pointer to an array to which the results are to be saved. This
 *             should be a pointer to the start of the array, and the 
 *             ``offset_ptr`` parameter should be used for aligning it with 
 *             the HODLR blocks.
 *             This array *must* be populated since the results are added to 
 *             it. The values not being set is an undefined behaviour.
 *             Must not overlap with any of the other pointers - doing so is 
 *             an undefined behaviour.
 * :param out_ld: Leading dimension of ``out``, i.e. the number of rows of the
 *                full array. Must be greater than or equal to the number of
 *                rows of the full HODLR matrix.
 * :param workspace: Pointer to an array that can be used as a workspace. 
 *                   Must be of at least size S x N where S is the number of 
 *                   retained singular values (the greater number between the
 *                   two off-diagonal nodes on ``parent``) and N is 
 *                   ``matrix_n``.
 *                   Must not overlap with any of the other arrays - doing 
 *                   so is an undefined behaviour.
 * :param offset: The current offset into ``matrix`` and ``out`` arrays.
 *
 * :return: The new offset.
 */
static inline int multiply_off_diagonal_transpose_dense(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict matrix,
  const int matrix_n,
  const int matrix_ld,
  double *restrict out,
  const int out_ld,
  double *restrict workspace,
  int offset
) {
  const double one = 1.0;
  
  const int offset2 = offset + parent->children[1].leaf->data.off_diagonal.m;

  multiply_low_rank_transpose_dense(
    &parent->children[2].leaf->data.off_diagonal, matrix + offset2, matrix_n,
    matrix_ld, one, workspace, out + offset, out_ld
  );

  multiply_low_rank_transpose_dense(
    &parent->children[1].leaf->data.off_diagonal, matrix + offset, matrix_n,
    matrix_ld, one, workspace, out + offset2, out_ld
  );

  return offset2 + parent->children[1].leaf->data.off_diagonal.n;
}


/**
 * Multiplies a tree HODLR matrix by a dense matrix as a dense matrix.
 *
 * Given a HODLR tree and a dense matrix array, computes the matrix-matrix 
 * multiplication and returns the result as a dense matrix.
 *
 * :param hodlr: Pointer to the HODLR tree to multiply. This must be a 
 *               fully constructed HODLR tree, including all the data being 
 *               filled in. If the data has not been assigned (e.g. by using
 *               :c:func:`dense_to_tree_hodlr`), it will lead to undefined
 *               behaviour.
 *               If ``NULL``, the function will immediately abort.
 * :param matrix: Pointer to an array storing the dense matrix to use for the
 *                multiplication. Must be of size ``matrix_ld`` x ``matrix_n``
 *                of which M x ``matrix_n`` submatrix will be used for the 
 *                multiplication (where M is the number of rows of ``hodlr``).
 *                Must be stored in column-major order.
 *                Must not overlap with ``out`` and must be occupied with 
 *                values - either will lead to undefined behaviour.
 *                If ``NULL``, the function will immediately abort.
 * :param matrix_n: The number of columns of ``matrix``.
 * :param matrix_ld: The leading dimension of ``matrix``, i.e. the number of 
 *                   rows of the full array. Must be greater than or equal to 
 *                   the number of rows of ``hodlr``.
 * :param out: Pointer to an array to be used for storing the results of the
 *             multiplication. Must be of size ``out_ld`` x ``matrix_n``. Will
 *             be stored in column-major order.
 *             Must not overlap with ``vector``, otherwise undefined behaviour
 *             will ensue, but may be both filled with value (which will be
 *             overwritten) or empty (i.e. just allocated).
 *             If ``NULL``, a new array is allocated.
 * :param out_ld: The leading dimension of ``out``, i.e. the number of rows of
 *                the full array. Must be greater than or equal to the number
 *                of rows of ``hodlr``, even if ``out == NULL``, in which case
 *                ``out`` will be allocated with size ``out_ld`` x 
 *                ``matrix_n``.
 * :return: The ``out`` array with the results of the matrix-matrix
 *          multiplication stored inside.
 */
double * multiply_hodlr_dense(const struct TreeHODLR *hodlr,
                              const double *restrict matrix,
                              const int matrix_n,
                              const int matrix_ld,
                              double *restrict out,
                              const int out_ld) {
  if (hodlr == NULL || matrix == NULL) {
    return NULL;
  }
  if (out == NULL) {
    out = malloc(out_ld * matrix_n * sizeof(double));
    if (out == NULL) {
      return NULL;
    }
  }

  int offset = 0, idx=0, m = 0;
  long n_parent_nodes = hodlr->len_work_queue;
  const double alpha = 1.0, beta = 0.0;

  int workspace_size = compute_multiply_hodlr_dense_workspace(hodlr, matrix_n);
  double *workspace = malloc(workspace_size * sizeof(double));

  struct HODLRInternalNode **queue = hodlr->work_queue;
  
  for (int i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (int j = 0; j < 2; j++) {
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
    offset = 0;

    for (int j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (int k = 0; k < 2; k++) {
        offset = multiply_off_diagonal_dense(
          queue[idx], matrix, matrix_n, matrix_ld, 
          out, out_ld, workspace, offset
        );

        idx += 1;
      }

      queue[j] = queue[2 * j + 1]->parent;
    }
  }

  multiply_off_diagonal_dense(
    hodlr->root, matrix, matrix_n, matrix_ld, out, out_ld, workspace, 0
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
 * :param internal: A pointer to the internal node representing a HODLR matrix
 *                  to multiply. Must not be NULL and must be correctly 
 *                  allocated and fully constructed - anything else is 
 *                  undefined.
 * :param height: The height of the HODLR matrix represented by ``internal``.
 *                This must correspond with the number of internal nodes 
 *                starting from ``internal`` (including) all the way to the 
 *                bottom of the tree.
 * :param matrix: A pointer to access an array containing the matrix to 
 *                multiply. Must not be ``NULL`` and must be large enough to 
 *                store the ``matrix_ld`` x ``matrix_n`` matrix.
 * :param matrix_n: The number of columns of ``matrix``.
 * :param matrix_ld: The leading dimension of ``matrix``.
 * :param queue: A pointer to access an array of pointers to internal nodes.
 *               This is a workspace array used to loop over the tree. Must 
 *               not be ``NULL``.
 * :param workspace: A pointer to access an array containing enough space 
 *                   to store an ``s`` x ``matrix_n`` matrix, where ``s`` is 
 *                   the largest number of singular values kept on any leaf 
 *                   node of the ``internal`` tree. Must not be ``NULL``.
 * :param workspace2: A pointer to access an array containing enough space to
 *                    store an ``m`` x ``matrix_n`` matrix, where ``m`` is the
 *                    number of rows of the largest block of the ``internal``
 *                    tree. Must not be ``NULL``.
 * :param out: A pointer to access an array to be used to save the results.
 *             Must be large enough to store a ``out_ld`` x ``matrix_n`` 
 *             matrix. Must not be ``NULL``.
 * :param out_ld: The leading dimension of ``out``.
 */
void multiply_internal_node_dense(
  const struct HODLRInternalNode *restrict const internal,
  const int height,
  const double *restrict const matrix,
  const int matrix_n,
  const int matrix_ld,
  const struct HODLRInternalNode **restrict queue,
  double *restrict workspace,
  double *restrict out,
  const int out_ld
) {
  int len_queue = 1, q_next_node_density = (int)pow(2, height-1);
  int q_current_node_density = q_next_node_density;
  int idx = 0, offset = 0;
  const double alpha = 1.0, beta = 0.0;

  int m = internal->children[1].leaf->data.off_diagonal.m;

  multiply_low_rank_dense(&internal->children[1].leaf->data.off_diagonal,
                          matrix + m, matrix_n, matrix_ld, beta,
                          workspace, out, out_ld);

  multiply_low_rank_dense(&internal->children[2].leaf->data.off_diagonal,
                          matrix, matrix_n, matrix_ld, beta,
                          workspace, out + m, out_ld);

  queue[0] = internal;
  for (int _ = 1; _ < height; _++) {
    q_next_node_density /= 2;
    offset = 0;

    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;
      for (int child = 0; child < 4; child += 3) {
        offset = multiply_off_diagonal_dense(
          queue[idx]->children[child].internal, matrix, matrix_n, matrix_ld, 
          out, out_ld, workspace, offset
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
      m = queue[node]->children[child].leaf->data.diagonal.m;
      dgemm_("N", "N", &m, &matrix_n, &m, &alpha, 
              queue[node]->children[child].leaf->data.diagonal.data, 
              &m, matrix + offset, &matrix_ld,
              &alpha, out + offset, &out_ld);
      offset += m;
    }
  }
}


/**
 * Multiplies the transpose of a tree HODLR matrix by a dense matrix as a 
 * dense matrix.
 *
 * Given a HODLR tree and a dense matrix array, transposes the HODLR tree,
 * computes the matrix-matrix multiplication and returns the result as a 
 * dense matrix.
 *
 * :param hodlr: Pointer to the HODLR tree to multiply. This must be a 
 *               fully constructed HODLR tree, including all the data being 
 *               filled in. If the data has not been assigned (e.g. by using
 *               :c:func:`dense_to_tree_hodlr`), it will lead to undefined
 *               behaviour.
 *               If ``NULL``, the function will immediately abort.
 * :param matrix: Pointer to an array storing the dense matrix to use for the
 *                multiplication. Must be of size ``matrix_ld`` x ``matrix_n``
 *                of which M x ``matrix_n`` submatrix will be used for the 
 *                multiplication (where M is the number of rows of ``hodlr``).
 *                Must be stored in column-major order.
 *                Must not overlap with ``out`` and must be occupied with 
 *                values - either will lead to undefined behaviour.
 *                If ``NULL``, the function will immediately abort.
 * :param matrix_n: The number of columns of ``matrix``.
 * :param matrix_ld: The leading dimension of ``matrix``, i.e. the number of 
 *                   rows of the full array. Must be greater than or equal to 
 *                   the number of rows of ``hodlr``.
 * :param out: Pointer to an array to be used for storing the results of the
 *             multiplication. Must be of size ``out_ld`` x ``matrix_n``. Will
 *             be stored in column-major order.
 *             Must not overlap with ``vector``, otherwise undefined behaviour
 *             will ensue, but may be both filled with value (which will be
 *             overwritten) or empty (i.e. just allocated).
 *             If ``NULL``, a new array is allocated.
 * :param out_ld: The leading dimension of ``out``, i.e. the number of rows of
 *                the full array. Must be greater than or equal to the number
 *                of rows of ``hodlr``, even if ``out == NULL``, in which case
 *                ``out`` will be allocated with size ``out_ld`` x 
 *                ``matrix_n``.
 * :return: The ``out`` array with the results of the matrix-matrix
 *          multiplication stored inside.
 */
double * multiply_hodlr_transpose_dense(const struct TreeHODLR *hodlr,
                                        const double *restrict matrix,
                                        const int matrix_n,
                                        const int matrix_ld,
                                        double *restrict out,
                                        const int out_ld) {
  if (hodlr == NULL || matrix == NULL) {
    return NULL;
  }
  if (out == NULL) {
    out = malloc(out_ld * matrix_n * sizeof(double));
    if (out == NULL) {
      return NULL;
    }
  }

  int workspace_size = compute_multiply_hodlr_dense_workspace(hodlr, matrix_n);
  double *workspace = malloc(workspace_size * sizeof(double));

  struct HODLRInternalNode **queue = hodlr->work_queue;

  multiply_internal_node_transpose_dense(
    hodlr->root, hodlr->height, matrix, matrix_n, matrix_ld, queue, workspace, 
    out, out_ld
  );

  return out;
}


/**
 * Multiplies the transpose of a HODLR matrix represented by an internal 
 * node and a dense matrix.
 *
 * Given an internal node and its height, and a dense matrix, transposes the
 * internal node and computes their product as a dense matrix.
 *
 * :param internal: A pointer to the internal node representing a HODLR matrix
 *                  to multiply. Must not be NULL and must be correctly 
 *                  allocated and fully constructed - anything else is 
 *                  undefined.
 * :param height: The height of the HODLR matrix represented by ``internal``.
 *                This must correspond with the number of internal nodes 
 *                starting from ``internal`` (including) all the way to the 
 *                bottom of the tree.
 * :param matrix: A pointer to access an array containing the matrix to 
 *                multiply. Must not be ``NULL`` and must be large enough to 
 *                store the ``matrix_ld`` x ``matrix_n`` matrix.
 * :param matrix_n: The number of columns of ``matrix``.
 * :param matrix_ld: The leading dimension of ``matrix``.
 * :param queue: A pointer to access an array of pointers to internal nodes.
 *               This is a workspace array used to loop over the tree. Must 
 *               not be ``NULL``.
 * :param workspace: A pointer to access an array containing enough space 
 *                   to store an ``s`` x ``matrix_n`` matrix, where ``s`` is 
 *                   the largest number of singular values kept on any leaf 
 *                   node of the ``internal`` tree. Must not be ``NULL``.
 * :param out: A pointer to access an array to be used to save the results.
 *             Must be large enough to store a ``out_ld`` x ``matrix_n`` 
 *             matrix. Must not be ``NULL``.
 * :param out_ld: The leading dimension of ``out``.
 */
void multiply_internal_node_transpose_dense(
  const struct HODLRInternalNode *restrict internal,
  const int height,
  const double *restrict matrix,
  const int matrix_n,
  const int matrix_ld,
  const struct HODLRInternalNode **restrict queue,
  double *restrict workspace,
  double *restrict out,
  const int out_ld
) {
  int len_queue = 1, q_next_node_density = (int)pow(2, height-1);
  int q_current_node_density = q_next_node_density;
  int idx = 0, offset = 0;
  const double alpha = 1.0, beta = 0.0;

  int m = internal->children[1].leaf->data.off_diagonal.m;

  multiply_low_rank_transpose_dense(
    &internal->children[2].leaf->data.off_diagonal, matrix + m, matrix_n, 
    matrix_ld, beta, workspace, out, out_ld
  );

  multiply_low_rank_transpose_dense(
    &internal->children[1].leaf->data.off_diagonal, matrix, matrix_n, 
    matrix_ld, beta, workspace, out + m, out_ld
  );

  queue[0] = internal;
  for (int _ = 1; _ < height; _++) {
    q_next_node_density /= 2;
    offset = 0;

    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;
      for (int child = 0; child < 4; child += 3) {
        offset = multiply_off_diagonal_transpose_dense(
          queue[idx]->children[child].internal,
          matrix, matrix_n, matrix_ld, out, out_ld, workspace, offset
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
      m = queue[node]->children[child].leaf->data.diagonal.m;
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
 * :param node: A pointer to the off-diagonal node to multiply. It must not be
 *              ``NULL`` and must point to a valid node with correctly 
 *              allocated and set values, anything else is undefined.
 * :param matrix: A pointer to an array containing the dense matrix to be used
 *                for the multiplication. Must not be ``NULL``. Must not 
 *                overlap with any other pointers.
 * :param matrix_m: The number of rows of ``matrix`` to multiply.
 * :param matrix_ld: The leading dimension of ``matrix``.
 * :param beta2: The value of ``beta`` to use for the final ``dgesdd``. 
 *               Determines whether results overwrite ``out`` or are added.
 * :param workspace: A pointer to a workspace array to be used. Must be large
 *                   enough to accomodate a ``s`` x ``matrix_n`` matrix where
 *                   ``s`` is the number of stored singular values on 
 *                   ``node``. Must not overlap with any other pointers.
 * :param out: A pointer to an array to which to store the result. Must be 
 *             large enough to accomodate a ``m`` x ``matrix_n`` matrix, where
 *             ``m`` is the number of rows of ``node``. Must not overlap with
 *             any other pointers.
 * :param out_ld: The leading dimension of ``out``.
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
 * Computes the matrix-matrix multiplication of a dense matrix with two 
 * off-diagonal blocks of a tree HODLR matrix internal node, and adds the 
 * results to the ``out`` matrix.
 *
 * :param parent: Pointer to an internal node holding the off-diagonal nodes
 *                to multiply.
 * :param matrix: Pointer to an array holding the matrix being multiplied.
 *                This should be a pointer to the start of the array, and the
 *                ``offset_ptr`` parameter should be used for aligning the 
 *                matrix and the HODLR blocks. Must be in column-major order.
 *                Must not overlap with any of the other pointers - doing so
 *                is an undefined behaviour.
 * :param matrix_m: The number of rows of ``matrix`` to multiply.
 * :param matrix_ld: The leading dimension of ``matrix``, i.e. the number of
 *                   rows of the full array. Must be greater or equal to the
 *                   number of rows of the full HODLR matrix.
 * :param out: Pointer to an array to which the results are to be saved. This
 *             should be a pointer to the start of the array, and the 
 *             ``offset_ptr`` parameter should be used for aligning it with 
 *             the HODLR blocks.
 *             This array *must* be populated since the results are added to 
 *             it. The values not being set is an undefined behaviour.
 *             Must not overlap with any of the other pointers - doing so is 
 *             an undefined behaviour.
 * :param out_ld: Leading dimension of ``out``, i.e. the number of rows of the
 *                full array. Must be greater than or equal to the number of
 *                rows of the full HODLR matrix.
 * :param workspace: Pointer to an array that can be used as a workspace. 
 *                   Must be of at least size S x N where S is the number of 
 *                   retained singular values (the greater number between the
 *                   two off-diagonal nodes on ``parent``) and N is 
 *                   ``matrix_n``.
 *                   Must not overlap with any of the other arrays - doing 
 *                   so is an undefined behaviour.
 * :param offset: The current offset into ``matrix`` and ``out`` arrays.
 *
 * :return: The new offset.
 */
static inline int multiply_dense_off_diagonal(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict matrix,
  const int matrix_m,
  const int matrix_ld,
  double *restrict out,
  const int out_ld,
  double *restrict workspace,
  const int offset
) {
  const double one = 1.0;
  const int offset2 = offset + parent->children[1].leaf->data.off_diagonal.m;

  multiply_dense_low_rank(
    &parent->children[1].leaf->data.off_diagonal, matrix + offset * matrix_ld, 
    matrix_m, matrix_ld, one, workspace, out + offset2 * out_ld, out_ld
  );

  multiply_dense_low_rank(
    &parent->children[2].leaf->data.off_diagonal, matrix + offset2 * matrix_ld, 
    matrix_m, matrix_ld, one, workspace, out + offset * out_ld, out_ld
  );

  return offset2 + parent->children[1].leaf->data.off_diagonal.n;
}


/**
 * Multiplies a dense matrix by a tree HODLR matrix as a dense matrix.
 *
 * Given a dense matrix and a HODLR tree, computes the matrix-matrix 
 * multiplication and returns the result as a dense matrix.
 *
 * :param hodlr: Pointer to the HODLR tree to multiply. This must be a 
 *               fully constructed HODLR tree, including all the data being 
 *               filled in. If the data has not been assigned (e.g. by using
 *               :c:func:`dense_to_tree_hodlr`), it will lead to undefined
 *               behaviour.
 *               If ``NULL``, the function will immediately abort.
 * :param matrix: Pointer to an array storing the dense matrix to use for the
 *                multiplication. Must be of size ``matrix_ld`` x ``matrix_n``
 *                of which M x ``matrix_n`` submatrix will be used for the 
 *                multiplication (where M is the number of rows of ``hodlr``).
 *                Must be stored in column-major order.
 *                Must not overlap with ``out`` and must be occupied with 
 *                values - either will lead to undefined behaviour.
 *                If ``NULL``, the function will immediately abort.
 * :param matrix_m: The number of rows of ``matrix`` to multiply.
 * :param matrix_ld: The leading dimension of ``matrix``, i.e. the number of 
 *                   rows of the full array. Must be greater than or equal to 
 *                   the number of rows of ``hodlr``.
 * :param out: Pointer to an array to be used for storing the results of the
 *             multiplication. Must be of size ``out_ld`` x ``matrix_n``. Will
 *             be stored in column-major order.
 *             Must not overlap with ``vector``, otherwise undefined behaviour
 *             will ensue, but may be both filled with value (which will be
 *             overwritten) or empty (i.e. just allocated).
 *             If ``NULL``, a new array is allocated.
 * :param out_ld: The leading dimension of ``out``, i.e. the number of rows of
 *                the full array. Must be greater than or equal to the number
 *                of rows of ``hodlr``, even if ``out == NULL``, in which case
 *                ``out`` will be allocated with size ``out_ld`` x 
 *                ``matrix_n``.
 * :return: The ``out`` array with the results of the matrix-matrix
 *          multiplication stored inside.
 */
double * multiply_dense_hodlr(const struct TreeHODLR *hodlr,
                              const double *restrict matrix,
                              const int matrix_m,
                              const int matrix_ld,
                              double *restrict out,
                              const int out_ld) {
  if (hodlr == NULL || matrix == NULL) {
    return NULL;
  }
  if (out == NULL) {
    out = malloc(out_ld * hodlr->root->m * sizeof(double));
    if (out == NULL) {
      return NULL;
    }
  }

  int offset = 0, idx=0, m = 0;
  long n_parent_nodes = hodlr->len_work_queue;
  const double alpha = 1.0, beta = 0.0;

  int workspace_size = compute_multiply_hodlr_dense_workspace(hodlr, matrix_m);
  double *workspace = malloc(workspace_size * sizeof(double));

  struct HODLRInternalNode **queue = hodlr->work_queue;
  
  for (int i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (int j = 0; j < 2; j++) {
      m = hodlr->innermost_leaves[idx]->data.diagonal.m;
      dgemm_("N", "N", &matrix_m, &m, &m, &alpha, 
             matrix + offset * matrix_ld, &matrix_ld,
             hodlr->innermost_leaves[idx]->data.diagonal.data, 
             &m, &beta, out + offset * out_ld, &out_ld);
      
      offset += m;
      idx++;
    }
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0; idx = 0;

    for (int j = 0; j < n_parent_nodes; j++) {
      for (int k = 0; k < 2; k++) {
        offset = multiply_dense_off_diagonal(
          queue[idx], matrix, matrix_m, matrix_ld, out, out_ld, workspace, 
          offset
        );

        idx++;
      }

      queue[j] = queue[2 * j + 1]->parent;
    }
  }

  multiply_dense_off_diagonal(
    hodlr->root, matrix, matrix_m, matrix_ld, out, out_ld, workspace, 0
  );

  free(workspace);
        
  return out;
}


/**
 * Multiplies a dense matryx by a HODLR matrix represented by an internal 
 * node.
 *
 * Given a dense matrix, and an internal node and its height, computes their
 * product as a dense matrix.
 *
 * :param internal: A pointer to the internal node representing a HODLR matrix
 *                  to multiply. Must not be NULL and must be correctly 
 *                  allocated and fully constructed - anything else is 
 *                  undefined.
 * :param height: The height of the HODLR matrix represented by ``internal``.
 *                This must correspond with the number of internal nodes 
 *                starting from ``internal`` (including) all the way to the 
 *                bottom of the tree.
 * :param matrix: A pointer to access an array containing the matrix to 
 *                multiply. Must not be ``NULL`` and must be large enough to 
 *                store the ``matrix_ld`` x ``matrix_n`` matrix.
 * :param matrix_m: The number of rows of ``matrix`` to multiply.
 * :param matrix_ld: The leading dimension of ``matrix``.
 * :param queue: A pointer to access an array of pointers to internal nodes.
 *               This is a workspace array used to loop over the tree. Must 
 *               not be ``NULL``.
 * :param workspace: A pointer to access an array containing enough space 
 *                   to store an ``s`` x ``matrix_n`` matrix, where ``s`` is 
 *                   the largest number of singular values kept on any leaf 
 *                   node of the ``internal`` tree. Must not be ``NULL``.
 * :param out: A pointer to access an array to be used to save the results.
 *             Must be large enough to store a ``out_ld`` x ``matrix_n`` 
 *             matrix. Must not be ``NULL``.
 * :param out_ld: The leading dimension of ``out``.
 */
void multiply_dense_internal_node(
  const struct HODLRInternalNode *restrict const internal,
  const int height,
  const double *restrict const matrix,
  const int matrix_m,
  const int matrix_ld,
  const struct HODLRInternalNode **restrict queue,
  double *restrict workspace,
  double *restrict out,
  const int out_ld
) {
  int len_queue = 1, q_next_node_density = (int)pow(2, height-1);
  int q_current_node_density = q_next_node_density;
  int idx = 0, offset = 0;
  const double alpha = 1.0, beta = 0.0;

  int m = internal->children[1].leaf->data.off_diagonal.m;

  multiply_dense_low_rank(&internal->children[1].leaf->data.off_diagonal,
                          matrix, matrix_m, matrix_ld, beta,
                          workspace, out + m * out_ld, out_ld);

  multiply_dense_low_rank(&internal->children[2].leaf->data.off_diagonal,
                          matrix + m * matrix_ld, matrix_m, matrix_ld, beta,
                          workspace, out, out_ld);

  queue[0] = internal;
  for (int _ = 1; _ < height; _++) {
    q_next_node_density /= 2;
    offset = 0;

    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;
      for (int child = 0; child < 4; child += 3) {
        offset = multiply_dense_off_diagonal(
          queue[idx]->children[child].internal,
          matrix, matrix_m, matrix_ld, out, out_ld, workspace, offset
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
      m = queue[node]->children[child].leaf->data.diagonal.m;
      dgemm_("N", "N", &matrix_m, &m, &m, &alpha, 
             matrix + offset * matrix_ld, &matrix_ld,
             queue[node]->children[child].leaf->data.diagonal.data, &m,
             &alpha, out + offset * out_ld, &out_ld);
      offset += m;
    }
  }
}

