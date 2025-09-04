#include <stdlib.h>

#include "../include/hmat_lib/hodlr.h"
#include "../include/hmat_lib/error.h"
#include "../include/hmat_lib/constructors.h"

#include "../include/internal/lapack_wrapper.h"
#include "../include/internal/blas_wrapper.h"


/**
 * Computes and sets the sizes of all HODLR internal nodes from the matrix 
 * size.
 *
 * Given a HODLR tree and the size of the matrix (number of rows), iterates 
 * over the entire tree and computes the size of each node's dimensions (``m``
 * and ``n`` as appropriate) by halving the parent's block size. The results 
 * are saved on the ``hodlr``.
 *
 * Sets the block sizes for all nodes - internal nodes, diagonal leaf nodes,
 * and off-diagonal leaf nodes, but does not set the ranks (``s``) for the
 * off-diagonal leaf nodes.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to the HODLR tree whose block sizes to set. Must be a 
 *     correctly allocated tree but should be empty - any data on the tree may 
 *     be overwritten. Must not be ``NULL``.
 * m
 *     The number of rows of the full HODLR matrix.
 *
 * Returns
 * -------
 * struct HODLRInternalNode **
 *     Pointer representing an array of pointers to internal nodes. This is
 *     the :c:member:`TreeHODLR.work_queue` but populated with the innermost 
 *     internal nodes on ``hodlr``.
 *
 * See Also
 * --------
 * compute_block_sizes_custom : Uses the sizes of the diagonal blocks.
 */
static struct HODLRInternalNode ** compute_block_sizes_halves(
  struct TreeHODLR *restrict const hodlr,
  const int m
) {
  struct HODLRInternalNode **queue = hodlr->work_queue;
  long len_queue = 1, q_next_node_density = hodlr->len_work_queue;
  long q_current_node_density = q_next_node_density;
  
  hodlr->root->m = m;
  queue[0] = hodlr->root;

  for (int _ = 1; _ < hodlr->height; _++) {
    q_next_node_density /= 2;
    for (int parent = 0; parent < len_queue; parent++) {
      const int idx = parent * q_current_node_density;
      
      const int m_smaller = queue[idx]->m / 2;
      const int m_larger = queue[idx]->m - m_smaller;

      queue[idx]->children[0].internal->m = m_larger;
      queue[idx]->children[3].internal->m = m_smaller;

      queue[idx]->children[1].leaf->data.off_diagonal.m = m_larger;
      queue[idx]->children[1].leaf->data.off_diagonal.n = m_smaller;

      queue[idx]->children[2].leaf->data.off_diagonal.m = m_smaller;
      queue[idx]->children[2].leaf->data.off_diagonal.n = m_larger;

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[idx]->children[3].internal;
      queue[idx] = queue[idx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  for (int parent = 0; parent < len_queue; parent++) {
    const int m_smaller = queue[parent]->m / 2;
    const int m_larger = queue[parent]->m - m_smaller;

    queue[parent]->children[0].leaf->data.diagonal.m = m_larger;
    queue[parent]->children[3].leaf->data.diagonal.m = m_smaller;

    queue[parent]->children[1].leaf->data.off_diagonal.m = m_larger;
    queue[parent]->children[1].leaf->data.off_diagonal.n = m_smaller;

    queue[parent]->children[2].leaf->data.off_diagonal.m = m_smaller;
    queue[parent]->children[2].leaf->data.off_diagonal.n = m_larger;
  }

  return queue;
}


/**
 * Computes and sets the sizes of all HODLR internal nodes given the sizes of 
 * dense blocks.
 *
 * Given a HODLR tree and the size (number of rows) of each dense block of the
 * matrix (the innermost diagonal blocks, i.e. 
 * :c:member:`TreeHODLR.innermost_leaves`), iterates over the entire tree and 
 * computes the size of each node's dimensions by summing the two children's
 * block sizes (or looking at the siblings' block sizes as appropriate). The 
 * results are saved on the ``hodlr``.
 *
 * Sets the block sizes for all nodes - internal nodes, diagonal leaf nodes,
 * and off-diagonal leaf nodes, but does not set the ranks (``s``) for the
 * off-diagonal leaf nodes.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to the HODLR tree whose block sizes to set. Must be a 
 *     correctly allocated tree but should be empty - any data on the tree may 
 *     be overwritten. Must not be ``NULL``.
 * m
 *     The number of rows of the full HODLR matrix.
 *
 * Returns
 * -------
 * struct HODLRInternalNode **
 *     Pointer representing an array of pointers to internal nodes. This is
 *     the :c:member:`TreeHODLR.work_queue` but populated with the innermost 
 *     internal nodes on ``hodlr``.
 *
 * See Also
 * --------
 * compute_block_sizes_halves : Uses the size of the full matrix.
 */
static struct HODLRInternalNode ** compute_block_sizes_custom(
  struct TreeHODLR *restrict const hodlr,
  const int *restrict const ms
) {
  struct HODLRInternalNode **queue = hodlr->work_queue;
  long n_parent_nodes = hodlr->len_work_queue;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    const int idx = 2 * parent;
    hodlr->innermost_leaves[idx]->data.diagonal.m = ms[idx];
    hodlr->innermost_leaves[idx + 1]->data.diagonal.m = ms[2 * parent + 1];

    queue[parent] = hodlr->innermost_leaves[idx]->parent;
    queue[parent]->m = ms[idx] + ms[2 * parent + 1];

    queue[parent]->children[1].leaf->data.off_diagonal.m = ms[idx];
    queue[parent]->children[1].leaf->data.off_diagonal.n = ms[idx + 1];

    queue[parent]->children[2].leaf->data.off_diagonal.m = ms[idx + 1];
    queue[parent]->children[2].leaf->data.off_diagonal.n = ms[idx];
  }

  for (int _ = hodlr->height - 1; _ > 0; _--) {
    n_parent_nodes /= 2;
    for (int parent = 0; parent < n_parent_nodes; parent++) {
      const int m1 = queue[2 * parent]->m, m2 =queue[2 * parent + 1]->m;
      queue[2 * parent]->parent->m = m1 + m2;

      queue[parent] = queue[2 * parent]->parent;

      queue[parent]->children[1].leaf->data.off_diagonal.m = m1;
      queue[parent]->children[1].leaf->data.off_diagonal.n = m2;

      queue[parent]->children[2].leaf->data.off_diagonal.m = m2;
      queue[parent]->children[2].leaf->data.off_diagonal.n = m1;
    }
  }

  for (int parent = 0; parent < hodlr->len_work_queue; parent++) {
    queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;
  }

  return queue;
}


/**
 * Copies each innermost diagonal leaf node data from a matrix.
 *
 * Given a dense matrix and an array of the innermost (higest depth) internal
 * nodes of a HODLR tree, copies the appropriate data from the dense matrix 
 * into the data array (:c:member:`NodeDiagonal.data`) of each node's two 
 * diagonal children.
 *
 * Parameters
 * ----------
 * matrix
 *     Pointer representing an array holding the dense matrix from which to 
 *     copy data. Must be an ``matrix_ld`` x ``matrix_ld`` square 2D 
 *     column-major matrix. Must not be ``NULL``.
 * matrix_ld
 *     The number of rows and columns of ``matrix``.
 * queue
 *     Pointer representing an array of pointers to internal nodes. Must be 
 *     of length ``len_queue`` and fully filled by the lowest-level internal
 *     nodes (i.e. the parents of :c:member:`TreeHODLR.innermost_leaves`).
 * len_queue
 *     The length of the ``len_queue`` array.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. An error status code from :c:enum:`ErrorCode` is written into 
 *     the pointer **if an error occurs**. Must not be ``NULL`` - doing so is 
 *     undefined.
 *
 * Errors
 * ------
 * ALLOCATION_FAILURE
 *     If one of the ``malloc`` calls fails. 
 */
static inline void copy_diagonal_blocks(
  const double *restrict const matrix,
  const int matrix_ld,
  struct HODLRInternalNode *restrict const *restrict queue,
  const long len_queue,
  int *restrict const ierr
#ifdef _TEST_HODLR
  , void *(*malloc)(size_t size)
#endif
) {
  int offset = 0;

  for (int parent = 0; parent < len_queue; parent++) {
    for (int child = 0; child < 4; child += 3) {
      const int m = queue[parent]->children[child].leaf->data.diagonal.m;

      double *data = malloc(m * m * sizeof(double));
      if (data == NULL) {
        *ierr = ALLOCATION_FAILURE;
        return;
      }
      for (int j = 0; j < m; j++) {
        for (int i = 0; i < m; i++) {
          data[i + j * m] = matrix[i + offset + (j + offset) * matrix_ld];
        }
      }
      queue[parent]->children[child].leaf->data.diagonal.data = data;

      offset += m;
    }
  }
}


/**
 * Compresses a single block of a matrix into an off-diagonal node using SVD.
 *
 * Computes the SVD of a dense matrix block and saves the significant parts of
 * the U and V matrices of an off-diagonal leaf node.
 *
 * Parameters
 * ----------
 * node
 *     The node on which to save the results. Must be allocated and have its 
 *     ``m`` and ``n`` values set. However, the ``u``, ``v``, and ``s`` fields
 *     should be empty as they will be overwritten.
 * m_smaller
 *     The smaller value between ``node->m`` and ``node->n``.
 * matrix_ld
 *     Leading dimension of ``lapack_matrix`` - he number of rows of the full 
 *     matrix must be greater than or equal to ``m_smaller``.
 * lapack_matrix
 *     Pointer representing an array containing the column-major 2D matrix to 
 *     compress. This may be a subset of a larger matrix - in this the 
 *     compressed block is ``lapack_matrix[:node->m, :node->n]``, i.e. values
 *     starting from the beginning of the passed-in array will be used and
 *     ``matrix_ld`` will be used to traverse the full matrix. 
 *     Might be overwritten by the SVD routine.
 * s
 *     Pointer representing an array used to temporarily store all the 
 *     singular values of ``lapack_matrix``. Must be a 1D aray of size of at 
 *     least ``m_smaller``.
 * u
 *     Pointer representing an array used to temporarily store the full U
 *     matrix. Must be of size of at least ``node->m`` x ``m_smaller``.
 * vt
 *     Pointer representing an array used to temporarily store the full
 *     :math:`V^T` matrix. Must be of size of at least ``m_smaller``  x 
 *     ``node->n``.
 * svd_threshold
 *     The threshold for discarding singular values after the SVD. Any 
 *     singular values smaller than one ``svd_threshold``-th of the first 
 *     singular value will be treated as approximately zero and therefore 
 *     the corresponding column vectors of the :math:`U` and :math:`V` 
 *     matrices will be discarded.
 * ierr
 *     Pointer to an integer hodling the :c:member:`ErrorCode.SUCCESS` value. 
 *     Used to signal the success or failure of this function. An error status 
 *     code from :c:enum:`ErrorCode` is written into the pointer 
 *     **if an error occurs**. Must not be ``NULL`` - doing so is undefined.
 *
 * Returns
 * -------
 * int
 *     The return code from the ``dgesdd`` routine.
 *
 * Errors
 * ------
 * ALLOCATION_FAILURE
 *     If one of the ``malloc`` calls in this function fails.
 * SVD_ALLOCATION_FAILURE
 *     If one of the ``malloc`` calls in the :c:func:`svd_double` function
 *     fails.
 * SVD_FAILURE
 *     If the ``dgesdd`` routine fails.
 */
static inline int compress_off_diagonal(
  struct NodeOffDiagonal *restrict const node,
  const int m_smaller,
  const int matrix_ld,
  double *restrict const lapack_matrix,
  double *restrict const s,
  double *restrict const u,
  double *restrict const vt,
  const double svd_threshold,
  int *restrict const ierr
#ifdef _TEST_HODLR
  , void *(*malloc)(size_t size)
#endif
) {
  const int m = node->m, n = node->n;
  int result = 
    svd_double(m, n, m_smaller, matrix_ld, lapack_matrix, s, u, vt, ierr);
  if (*ierr != SUCCESS) {
    return result;
  }

  int svd_cutoff_idx = 1;
  if (s[0] > svd_threshold) {
    for (svd_cutoff_idx=1; svd_cutoff_idx < m_smaller; svd_cutoff_idx++) {
      if (s[svd_cutoff_idx] < svd_threshold * s[0]) {
        break;
      }
    }
  }

  double *u_top_right = malloc(m * svd_cutoff_idx * sizeof(double));
  if (u_top_right == NULL) {
    #pragma omp atomic write
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<m; j++) {
      u_top_right[j + i * m] = u[j + i * m] * s[i];
    }
  }

  double *v_store = malloc(svd_cutoff_idx * n * sizeof(double));
  if (v_store == NULL) {
    #pragma omp atomic write
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<n; j++) {
      v_store[j + i * n] = vt[i + j * m_smaller];
    }
  }

  node->u = u_top_right;
  node->v = v_store;
  node->s = svd_cutoff_idx;

  return result;
}


/**
 * Compresses an entire dense matrix into the HODLR format using SVD.
 *
 * Given a HODLR tree and a dense matrix, iteratively compresses the matrix 
 * into the pre-allocated HODLR tree by compressing each block of the matrix 
 * into an off-diagona leaf node.
 *
 * This function contains OpenMP pragmas that schedule tasks - it assumes that
 * the function is being run from::
 *
 *    #pragma omp parallel
 *    #pragma omp single
 *    #pragma omp taskgroup
 *
 * If any failure occurs, the function exits early without cleaning up, 
 * setting ``ierr`` and returning an error code. When running with OpenMP,
 * an immediate abort is performed if cancelling is turned on, otherwise the
 * remaining work runs to completion.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to the HODLR into which to compress the ``matrix``. This tree
 *     must be fully and correctly allocated, and the block sizes of all nodes
 *     must already be set. However, all other data fields should be empty as
 *     they will be overwritten.
 * queue
 *     Pointer representing an array of pointers to internal nodes. Must be 
 *     fully filled with all the lowest-level internal nodes of ``hodlr`` 
 *     (i.e. the parents of :c:member:`TreeHODLR.innermost_leaves`). Must not
 *     be ``NULL``.
 * matrix
 *     Pointer representing an array holding the dense matrix to compress. 
 *     Must be an ``m`` x ``m`` square 2D column-major matrix. Must not be 
 *     ``NULL``.
 * matrix_ld
 *     The leading dimension of ``matrix``.
 * s
 *     Pointer representing array used as a workspace for storing the SVD 
 *     singular values. Must be large enough to accomodate all the singular 
 *     values for all HODLR compressions in parallel, i.e. be at least of size 
 *     ``4 * floor(matrix_ld / 2)``. Must not be ``NULL``.
 * u
 *     Pointer representing an array used as a workspace for storing the SVD U 
 *     matrices. Must be large enough to ccomodate all the U matrices for all 
 *     HODLR compressions, i.e. be at least of size 
 *     ``4 * floor(m / 2) * ceil(m / 2)``. Must noe be ``NULL``.
 * vt
 *     Pointer representing an array used as a workspace for storing the SVD 
 *     :math:`V^T` matrices. Must be large enoug to accomodate all the 
 *     :math:`V^T` matrices for all HODLR compressions, i.e. be at least of 
 *     size ``4 * floor(m / 2) * ceil(m / 2)``. Must not be ``NULL``.
 * svd_threshold
 *     The threshold for discarding singular values after the SVD. Any 
 *     singular values smaller than one ``svd_threshold``-th of the first 
 *     singular value will be treated as approximately zero and therefore 
 *     the corresponding column vectors of the :math:`U` and :math:`V` 
 *     matrices will be discarded.
 * ierr
 *     Pointer to an integer hodling the :c:member:`ErrorCode.SUCCESS` value. 
 *     Used to signal the success or failure of this function. An error status 
 *     code from :c:enum:`ErrorCode` is written into the pointer 
 *     **if an error occurs**. Must not be ``NULL`` - doing so is undefined.
 *
 * Returns
 * -------
 * int
 *     The return code from the ``dgesdd`` routine.
 *
 * Errors
 * ------
 * ALLOCATION_FAILURE
 *     If one of the ``malloc`` calls in this function fails.
 * SVD_ALLOCATION_FAILURE
 *     If one of the ``malloc`` calls in the :c:func:`svd_double` function
 *     fails.
 * SVD_FAILURE
 *     If the ``dgesdd`` routine fails.
 */
static inline int compress_matrix(
  struct TreeHODLR *restrict const hodlr,
  struct HODLRInternalNode *restrict *restrict queue,
  double *restrict const matrix,
  const int matrix_ld,
  double *restrict const s,
  double *restrict const u,
  double *restrict const vt,
  const double svd_threshold,
  int *restrict const ierr
#ifdef _TEST_HODLR
  , void *(*malloc)(size_t size)
#endif
) {
  long n_parent_nodes = hodlr->len_work_queue;
  int offset_matrix = 0, offset_s = 0, offset_u = 0;
  struct NodeOffDiagonal *node = NULL;
  int result = 0, final_result = 0;

  for (int _ = hodlr->height; _ > 0; _--) {
    offset_matrix = 0;
  
    for (int parent = 0; parent < n_parent_nodes; parent++) {
      node = &(queue[parent]->children[1].leaf->data.off_diagonal);
      const int m = node->m, n = node->n;
      const int m_smaller = (m < n) ? m : n;

      double *sub_matrix_pointer = 
        matrix + offset_matrix + matrix_ld * (offset_matrix + m);

#ifndef _TEST_HODLR
#pragma omp task default(none) private(result) firstprivate(node, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, matrix_ld)
#else
#pragma omp task default(none) private(result) firstprivate(node, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, matrix_ld, malloc)
#endif
      {
        result = compress_off_diagonal(
          node, m_smaller, matrix_ld, sub_matrix_pointer,
          s + offset_s, u + offset_u, vt + offset_u, svd_threshold, ierr
#ifdef _TEST_HODLR
          , malloc
#endif
        );

        if (*ierr != SUCCESS) {
          //handle_error(ierr, result);
          #pragma omp atomic write
          final_result = result;

          #pragma omp cancel taskgroup

          #if !defined(_OPENMP)
          return result;
          #endif
        }
      }
      offset_s += m_smaller; offset_u += m * n;
  
      // Off-diagonal block in the bottom left corner
      sub_matrix_pointer = matrix + matrix_ld * offset_matrix + offset_matrix + m;
      node = &(queue[parent]->children[2].leaf->data.off_diagonal);

#ifndef _TEST_HODLR
#pragma omp task default(none) private(result) firstprivate(node, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, matrix_ld)
#else
#pragma omp task default(none) private(result) firstprivate(node, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, matrix_ld, malloc)
#endif
      {
        result = compress_off_diagonal(
          node, m_smaller, matrix_ld, sub_matrix_pointer, 
          s + offset_s, u + offset_u, vt + offset_u, svd_threshold, ierr
#ifdef _TEST_HODLR
          , malloc
#endif
        );
        if (*ierr != SUCCESS) {
          // error out
          #pragma omp atomic write
          final_result = result;
          #pragma omp cancel taskgroup

          #if !defined(_OPENMP)
          return result;
          #endif
        }
      }

      offset_s += m_smaller; offset_u += m * n;

      offset_matrix += m + n;

      queue[parent / 2] = queue[parent]->parent;
    }
    n_parent_nodes /= 2;
  }

  return final_result;
}


/**
 * Compresses a dense matrix into the HODLR format.
 *
 * Given an empty HODLR tree and a dense matrix, fills in the HODLR by 
 * compressing the off-diagonal blocks using SVD.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to a HODLR into which to compress ``matrix``. 
 *     Must be a fully allocated HODLR tree. Passing in a partially or 
 *     incorrectly allocated tree leads to undefined behaviour.
 *     *Should* be empty, with no data filled in, since all data will be 
 *     overwritten, potentially causing memory leaks.
 *     If ``NULL``, aborts immediately.
 * m
 *     The size of the ``matrix`` matrix. I.e., the number of rows and columns
 * ms
 *     (optional) Pointer representing an array containing the sizes of the 
 *     dense diagonal blocks to use for ``hodlr``. If provided, this allows 
 *     the HODLR tree to be split into blocks of custom sizes. If ``NULL``, 
 *     this parameter is ignored and the HODLR is split in halves.
 *     If provided, this must be an array of length :math:`2^h` where ``h`` is 
 *     the height of ``hodlr`` whose values add up to ``m``. A shorter array 
 *     leads to undefined behaviour. Each entry in the array must specify the 
 *     size of a dense diagonal block, starting with the one in the top left 
 *     corner (i.e. ``matrix[0]``).
 * matrix
 *     Pointer representing an array storing the dense matrix to compress. 
 *     This must be a square column-major 2D matrix of size ``m`` x ``m``.
 *     If ``NULL``, aborts immediately
 *     *Some of the values may be overwritten during SVD compuration.*
 * svd_threshold
 *     The threshold for discarding singular values after the SVD. Any 
 *     singular values smaller than one ``svd_threshold``-th of the first 
 *     singular value will be treated as approximately zero and therefore 
 *     the corresponding column vectors of the :math:`U` and :math:`V` 
 *     matrices will be discarded.
 * ierr
 *     Pointer to an integer hodling used to signal the success or failure of 
 *     this function. An status code from :c:enum:`ErrorCode` is written into 
 *     the pointer. Must not be ``NULL`` - doing so is undefined.
 *
 * Returns
 * -------
 * int
 *     The error code from the SVD routine.
 *
 * Errors
 * ------
 * INPUT_ERROR
 *     If ``hodlr`` or ``matrix`` is ``NULL`` or if ``ms`` does not match 
 *     ``m``.
 * ALLOCATION_FAILURE
 *     If one of the ``malloc`` calls in this function fails.
 * SVD_ALLOCATION_FAILURE
 *     If one of the ``malloc`` calls in the :c:func:`svd_double` function
 *     fails.
 * SVD_FAILURE
 *     If the ``dgesdd`` routine fails.
 *
 * Warnings
 * --------
 * If anything fails for any reason, the routine immediately aborts, setting
 * ``ierr`` and returning an error code as appropriate. No clean-up is 
 * performed, so it is likely that the data will be partially allocated. If 
 * that is the case, it must be freed (e.g. using :c:func:`free_tree_data`) 
 * before attempting to use this function again to avoid memory leaks.
 *
 * See Also
 * --------
 * allocate_tree : Used for allocating the tree structure
 *
 * Notes
 * -----
 * The compression is performed by first determining the size of each block of
 * the matrix, then copying the diagonal blocks into new, small dense arrays,
 * and lastly compressing each off-diagonal block using SVD and storing a 
 * number of columns of the U and V matrices obtained from SVD.
 */
int dense_to_tree_hodlr(
  struct TreeHODLR *restrict const hodlr, 
  const int m,
  const int *restrict const ms,
  double *restrict const matrix, 
  const double svd_threshold,
  int *restrict const ierr
#ifdef _TEST_HODLR
  , void *(*malloc)(size_t size),
  void(*free)(void *ptr)
#endif
) {
  if (hodlr == NULL || matrix == NULL) {
    *ierr = INPUT_ERROR;
    return 0;
  }
  *ierr = SUCCESS;

  struct HODLRInternalNode **queue;
  if (ms == NULL) {
    queue = compute_block_sizes_halves(hodlr, m);
  } else {
    queue = compute_block_sizes_custom(hodlr, ms);
    if (hodlr->root->m != m) {
      *ierr = INPUT_ERROR;
      return 0;
    }
  }

#ifndef _TEST_HODLR
  copy_diagonal_blocks(matrix, m, queue, hodlr->len_work_queue, ierr);
#else
  copy_diagonal_blocks(matrix, m, queue, hodlr->len_work_queue, ierr, malloc);
#endif

  if (*ierr != SUCCESS) {
    return 0;
  }

  const int m_larger = hodlr->root->children[1].leaf->data.off_diagonal.m;
  const int m_smaller = hodlr->root->children[1].leaf->data.off_diagonal.n;

  double *s = malloc(hodlr->height * m * sizeof(double));
  if (s == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return 0;
  }
  double *u = malloc(8 * m_larger * m_smaller * sizeof(double));
  if (u == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(s);
    return 0;
  }
  double *vt = u + (4 * m_larger * m_smaller);

  int result = 0;
  #pragma omp parallel
  {
    #pragma omp single
    {
      #pragma omp taskgroup
      {
        result = compress_matrix(
          hodlr, queue, matrix, m, s, u, vt, svd_threshold, ierr
#ifdef _TEST_HODLR
          , malloc
#endif
        );
      }
    }
  }

  free(s); free(u);
  
  return result;
}

