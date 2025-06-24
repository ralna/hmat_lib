#include <stdio.h>
#include <stdlib.h>

#include "../include/lapack_wrapper.h"
#include "../include/tree.h"
#include "../include/error.h"


static void print_matrix(int m, int n, double *matrix, int lda) {
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      printf("%f    ", matrix[j * lda + i]);
    }
    printf("\n");
  }
  printf("\n");
}


/**
 * Computes and sets the sizes of all HODLR internal nodes from the matrix 
 * size.
 *
 * Given a HODLR tree and the size of the matrix (number of rows), iterates 
 * over the entire tree and computes the size of each diagonal HODLR block
 * represented by a :c:struct:`HODLRInternalNode`. Saves each size on the 
 * appropriate node.
 *
 * :param hodlr: The HODLR tree whose block sizes to compute. Must be a 
 *               correctly allocated tree and should be empty - any data on 
 *               the tree may be overwritten. Must not be NULL; otherwise is
 *               undefined.
 * :param m: The number of rows of the full HODLR matrix.
 *
 * :return: A pointer to access an array of internal nodes which contains 
 *          the innermost internal nodes on ``hodlr``.
 */
static struct HODLRInternalNode ** compute_block_sizes_halves(
  struct TreeHODLR *restrict hodlr,
  const int m
) {
  struct HODLRInternalNode **queue = hodlr->work_queue;
  long len_queue = 1, q_next_node_density = hodlr->len_work_queue;
  long q_current_node_density = q_next_node_density;
  int m_smaller = 0, m_larger = 0, idx = 0;
  
  hodlr->root->m = m;
  queue[0] = hodlr->root;

  for (int _ = 1; _ < hodlr->height; _++) {
    q_next_node_density /= 2;
    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;
      
      m_smaller = queue[idx]->m / 2;
      m_larger = queue[idx]->m - m_smaller;

      queue[idx]->children[0].internal->m = m_larger;
      queue[idx]->children[3].internal->m = m_smaller;

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[idx]->children[3].internal;
      queue[idx] = queue[idx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  return queue;
}


/**
 * Computes and sets the sizes of all HODLR internal nodes given the sizes of 
 * dense blocks.
 *
 * Given a HODLR tree and the size (number of rows) of each dense block of the
 * matrix (the innermost diagonal blocks, i.e. 
 * :c:param:`TreeHODLR.innermost_leaves`), iterates over the entire tree and 
 * computes the size of each diagonal HODLR block node. Saves each size on the 
 * appropriate node.
 *
 * :param hodlr: Pointer to the HODLR tree whose block sizes to compute. Must 
 *               be a correctly allocated tree and should be empty - any data 
 *               on the tree may be overwritten. Must not be NULL; otherwise 
 *               is undefined.
 * :param ms: Pointer to access an array containing the number of rows of each 
 *            :c:struct:`NodeDiagonal` that will make up the HODLR tree.
 *
 * :return: A pointer to access an array of internal nodes which contains 
 *          the innermost internal nodes on ``hodlr``.
 */
static struct HODLRInternalNode ** compute_block_sizes_custom(
  struct TreeHODLR *hodlr,
  const int *ms
) {
  struct HODLRInternalNode **queue = hodlr->work_queue;
  long n_parent_nodes = hodlr->len_work_queue;
  int m_smaller = 0, m_larger = 0, idx = 0;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;
    queue[parent]->m = ms[2 * parent] + ms[2 * parent + 1];
  }

  for (int _ = hodlr->height - 1; _ > 0; _--) {
    n_parent_nodes /= 2;
    for (int parent = 0; parent < n_parent_nodes; parent++) {
      queue[2 * parent]->parent->m = queue[2 * parent]->m 
                                   + queue[2 * parent + 1]->m;
      queue[parent] = queue[2 * parent]->parent;
    }
  }

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;
  }

  return queue;
}


/**
 * Copies each innermost diagonal leaf node data from a matrix.
 *
 * Given a dense matrix and an array of the innermost (higest depth) internal
 * nodes of a HODLR tree, copies the appropriate data from the dense matrix 
 * into data array (:c:member:`NodeDiagonal.data`) of each node's two diagonal
 * children.
 *
 * :param matrix: Pointer to the array holding the dense matrix from which to 
 *                copy data. Must be an ``m`` x ``m`` square 2D column-major 
 *                matrix. Must not be NULL; otherwise is undefined.
 * :param m: The number of rows and columns of ``matrix``.
 * :param queue: Pointer to an array of internal nodes. Must be of length 
 *               ``n_parent_nodes``
 */
static void copy_diagonal_blocks(double *restrict matrix,
                                 int m,
                                 struct HODLRInternalNode **restrict queue,
                                 long n_parent_nodes,
                                 int *restrict ierr
#ifdef _TEST_HODLR
                                 , void *(*malloc)(size_t size)
#endif
                                 ) {
  int m_larger = 0, m_smaller = 0, offset = 0;
  double *data = NULL;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    m_smaller = queue[parent]->m / 2;
    m_larger = queue[parent]->m - m_smaller;

    data = malloc(m_larger * m_larger * sizeof(double));
    if (data == NULL) {
      *ierr = ALLOCATION_FAILURE;
      return;
    }
    for (int j = 0; j < m_larger; j++) {
      for (int i = 0; i < m_larger; i++) {
        data[i + j * m_larger] = matrix[i + offset + (j + offset) * m];
      }
    }
    queue[parent]->children[0].leaf->data.diagonal.data = data;
    queue[parent]->children[0].leaf->data.diagonal.m = m_larger;

    offset += m_larger;

    data = malloc(m_smaller * m_smaller * sizeof(double));
    if (data == NULL) {
      *ierr = ALLOCATION_FAILURE;
      return;
    }
    for (int j = 0; j < m_smaller; j++) {
      for (int i = 0; i < m_smaller; i++) {
        data[i + j * m_smaller] = matrix[i + offset + (j + offset) * m];
      }
    }
    queue[parent]->children[3].leaf->data.diagonal.data = data;
    queue[parent]->children[3].leaf->data.diagonal.m = m_smaller;

    offset += m_smaller;
  }
}


/**
 * Compresses a dense off-diagonal block.
 *
 * Internal function that computes the SVD of a matrix block
 * and saves the the significant parts of the U and V matrices
 * on a off-diagonal node.
 *
 * :param node: The node to which to save the results.
 * :param m: The number of rows of the ``lapack_matrix`` matrix.
 * :param n: The number of columns of the ``lapack_matrix`` matrix.
 * :param n_singular_values: The number of singular values returned
 *                           by the compact SVD. I.e. the smaller
 *                           value between ``m`` and ``n``.
 * :param matrix_leading_dim: The number of rows of the full matrix
 *                            ``lapack_matrix`` (lda). 
 *                            ``matrix_leading_dim`` >= ``m``.
 * :param lapack_matrix: A pointer to a column-major array containing
 *                       the 2D matrix to compress. This may be a 
 *                       subset of a larger matrix. Might be 
 *                       overwritten by the SVD routine.
 * :param s: Pointer to an array used to store all the singular values
 *           of ``lapack_matrix``. A 1D aray of size of at least
 *           ``n_singular_values``.
 * :param u: Pointer to an array used to temporarily store all the 
 *           columns of the U matrix of the ``lapack_matrix``. 
 *           A 2D array of size of at least ``m`` x 
 *           ``n_singular_values``.
 * :param vt: Pointer to an array used to temporarily store all the
 *            rows of the VT matrix of the ``lapack_matrix``.
 *            A 2D array of size of at least ``n_singular_values``
 *            x ``n``.
 * :param svd_threshold: The threshold for discarding singular values. 
 *                       Any singular values (and the corresponding 
 *                       column vectors of the U and V matrices) that 
 *                       satisfy :math:`s_i < t * s_0` will be 
 *                       discarded (:math:`s_i` is i-th singular value, 
 *                       :math:`t` is the ``svd_threshold``, and 
 *                       :math:`s_0` is the largest singular value).
 * :param ierr: Error code corresponding to :c:enum:`ErrorCode`. On 
 *              successful completion of the function, 
 *              :c:enum:`ErrorCode.SUCCESS` is returned. Otherwise,
 *              a corresponding error code is set.
 *              Must NOT be ``NULL`` pointer - passing in ``NULL``
 *              as ``ierr`` is undefined behaviour.
 * :return: The return code from the SVD routine.
 */
static int compress_off_diagonal(struct NodeOffDiagonal *restrict node,
                                 const int m, 
                                 const int n, 
                                 const int n_singular_values,
                                 const int matrix_leading_dim,
                                 double *restrict lapack_matrix,
                                 double *restrict s,
                                 double *restrict u,
                                 double *restrict vt,
                                 const double svd_threshold,
                                 int *restrict ierr
#ifdef _TEST_HODLR
                                 , void *(*malloc)(size_t size)
#endif
                                 ) {
  //printf("m=%d, n=%d, nsv=%d, lda=%d\n", m, n, n_singular_values, matrix_leading_dim);
  //print_matrix(matrix_leading_dim, matrix_leading_dim, lapack_matrix - 5);
  int result = svd_double(m, n, n_singular_values, matrix_leading_dim, 
                          lapack_matrix, s, u, vt, ierr);
  //printf("svd result %d\n", result);
  if (*ierr != SUCCESS) {
    return result;
  }

  int svd_cutoff_idx = 1;
  for (svd_cutoff_idx=1; svd_cutoff_idx < n_singular_values; svd_cutoff_idx++) {
    //printf("%f    ", s[svd_cutoff_idx]);
    if (s[svd_cutoff_idx] < svd_threshold * s[0]) {
      break;
    }
  }
  //printf("svd cut-off=%d, m=%d\n", svd_cutoff_idx, m);

  double *u_top_right = malloc(m * svd_cutoff_idx * sizeof(double));
  if (u_top_right == NULL) {
    #pragma omp atomic write
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<m; j++) {
      //printf("i=%d, j=%d, idx=%d\n", i, j, j + i * m);
      u_top_right[j + i * m] = u[j + i * m] * s[i];
    }
  }
  //print_matrix(svd_cutoff_idx, m, u_top_right);

  double *v_store = malloc(svd_cutoff_idx * n * sizeof(double));
  if (v_store == NULL) {
    #pragma omp atomic write
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<n; j++) {
      v_store[j + i * n] = vt[i + j * n_singular_values];
    }
  }
  //print_matrix(n, svd_cutoff_idx, v_store);

  node->u = u_top_right;
  node->v = v_store;

  node->m = m;
  node->s = svd_cutoff_idx;
  node->n = n;

  return result;
}


/**
 * Compresses a dense matrix into the HODLR format.
 *
 * Given a HODLR tree and a dense matrix, iteratively compresses the matrix 
 * into the pre-allocated HODLR tree by looping bottom-up starting with the 
 * bottom-most internal nodes in ``queue``.
 *
 * This function contains OpenMP pragmas that schedule tasks - it assumes that
 * the function is being run from:
 *
 * .. literal::
 * 
 *    #pragma omp parallel
 *    #pragma omp single
 *    #pragma omp taskgroup
 *
 * :param hodlr: The tree HODLR to which to compress the ``matrix``. This tree
 *               must be fully and correctly allocated, and the sizes of the
 *               internal nodes (i.e. :c:member:`HODLRInternalNode.m`) must 
 *               already be set.
 * :param queue: An array of pointers to internal nodes. Must be fully filled
 *               with all the lowest-level internal nodes of ``hodlr``.
 * :param matrix: Pointer to the array holding the dense matrix from which to 
 *                copy data. Must be an ``m`` x ``m`` square 2D column-major 
 *                matrix. Must not be NULL; otherwise is undefined.
 * :param m: The size the full ``matrix``.
 * :param s: Workspace for the SVD singular values. Must be large enough to 
 *           accomodate all the singular values for all HODLR compressions,
 *           i.e. be at least of size ``4 * floor(m / 2)``. NULL leads to 
 *           undefined behaviour.
 * :param u: Workspace for the SVD U matrices. Must be large enough to 
 *           accomodate all the U matrices for all HODLR compressions,
 *           i.e. be at least of size ``4 * floor(m / 2) * ceil(m / 2)``. NULL
 *           leads to undefined behaviour.
 * :param vt: Workspace for the SVD V^T matrices. Must be large enoug to 
 *            accomodate all the V^T matrices for all HODLR compressions, i.e.
 *            be at least of size ``4 * floor(m / 2) * ceil(m / 2)``. NULL 
 *            leads to undefined behaviour.
 * :param svd_threshold: The threshold for discarding singular values. Any 
 *                       singular values (and the corresponding column vectors 
 *                       of the U and V matrices) that satisfy 
 *                       :math:`s_i < t * s_0` will be discarded (:math:`s_i` 
 *                       is i-th singular value, :math:`t` is the 
 *                       ``svd_threshold``, and :math:`s_0` is the largest 
 *                       singular value).
 * :param ierr: Error code corresponding to :c:enum:`ErrorCode`. On any 
 *              failure, a corresponding error code is written to the pointer,
 *              but ``SUCCESS`` is not written here. NULL is undefined 
 *              behaviour.
 */
static int compress_matrix(struct TreeHODLR *restrict hodlr,
                           struct HODLRInternalNode **restrict queue,
                           double *restrict matrix,
                           const int m,
                           double *restrict s,
                           double *restrict u,
                           double *restrict vt,
                           const double svd_threshold,
                           int *ierr
#ifdef _TEST_HODLR
                           , void *(*malloc)(size_t size)
#endif
                           ) {
  long n_parent_nodes = hodlr->len_work_queue;
  int offset_matrix = 0, offset_s = 0, offset_u = 0;
  int m_smaller = 0, m_larger = 0;
  double *sub_matrix_pointer = NULL;
  struct NodeOffDiagonal *node = NULL;
  int result = 0, final_result = 0;

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset_matrix = 0;
  
    for (int parent = 0; parent < n_parent_nodes; parent++) {
      for (int child = 2 * parent; child < 2 * parent + 2; child++) {
        m_smaller = queue[child]->m / 2;
        m_larger = queue[child]->m - m_smaller;

        sub_matrix_pointer = matrix + offset_matrix + m * (offset_matrix + m_larger);
        node = &(queue[child]->children[1].leaf->data.off_diagonal);

#ifndef _TEST_HODLR
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m)
#else
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m, malloc)
#endif
        {
          result = compress_off_diagonal(
            node, 
            m_larger, m_smaller, m_smaller, m, sub_matrix_pointer,
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
        offset_s += m_smaller; offset_u += m_larger * m_smaller;
    
        // Off-diagonal block in the bottom left corner
        sub_matrix_pointer = matrix + m * offset_matrix + offset_matrix + m_larger;
        node = &(queue[child]->children[2].leaf->data.off_diagonal);

#ifndef _TEST_HODLR
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m)
#else
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m, malloc)
#endif
        {
          result = compress_off_diagonal(
            node, 
            m_smaller, m_larger, m_smaller, m, sub_matrix_pointer, 
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

        offset_s += m_larger; offset_u += m_larger * m_smaller;

        offset_matrix += m_larger + m_smaller;
      }

      queue[parent] = queue[2 * parent + 1]->parent;
    }
  }

  m_smaller = queue[0]->m / 2;
  m_larger = queue[0]->m - m_smaller;

  sub_matrix_pointer = matrix + m * m_larger;
  node = &(queue[0]->children[1].leaf->data.off_diagonal); 
#ifndef _TEST_HODLR
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m)
#else
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m, malloc)
#endif
  {
    result = compress_off_diagonal(
      node, 
      m_larger, m_smaller, m_smaller, m, sub_matrix_pointer,
      s + offset_s, u + offset_u, vt + offset_u, svd_threshold, ierr
#ifdef _TEST_HODLR
      , malloc
#endif
    );
    if (*ierr != SUCCESS) {
      //handle_error(ierr, result);  // 
      #pragma omp atomic write
      final_result = result;
      #pragma omp cancel taskgroup
            
      #if !defined(_OPENMP)
      return result;
      #endif
    }
  }
  offset_s += m_smaller; offset_u += m_larger * m_smaller;

  // Off-diagonal block in the bottom left corner
  sub_matrix_pointer = matrix + m_larger;
  node = &(queue[0]->children[2].leaf->data.off_diagonal);

#ifndef _TEST_HODLR
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m)
#else
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m, malloc)
#endif
  {
    result = compress_off_diagonal(
      node, 
      m_smaller, m_larger, m_smaller, m, sub_matrix_pointer, 
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

  return final_result;
}


/**
 * Compresses a dense matrix into a HODLR tree matrix.
 *
 * Requires a preallocated HODLR tree (e.g. from 
 * :c:func:`allocate_tree`), a matrix, and some settings.
 *
 * :param hodlr: A pointer to a HODLR tree. This *should* be an empty tree, 
 *               with no data filled in, since any pointers to data will be 
 *               overwritten with newly allocated data and therefore lost, 
 *               potentially causing memory leaks.
 *               If ``NULL``, aborts immediately. Passing in a partially or 
 *               incorrectly allocated tree is an undefined behaviour.
 *
 * :param m: The size of the ``matrix`` matrix. I.e., the number of rows and 
 *           columns.
 *
 * :param ms: (optional) A pointer to access an array containing the sizes 
 *            of the dense diagonal blocks. This allows the HODLR tree to be 
 *            split in a customised fashion. If ``NULL``, this parameter is 
 *            ignored and the HODLR is split in halves.
 *            If provided, this must be an array of length :math:`2^h` where
 *            ``h`` is the height of ``hodlr`` (shorter array leads to 
 *            undefined behaviour). Each entry in the array must specify the 
 *            size of a dense diagonal block, starting with the one in the 
 *            top left corner (i.e. ``matrix[0]``).
 *
 * :param matrix: The dense matrix to compress. This must be an column-major 
 *                array holding an ``m`` x ``m`` square 2D matrix.
 *                If ``NULL``, aborts immediately *Some of the values may be 
 *                overwritten during SVD compuration*
 *
 * :param svd_threshold: The threshold for discarding singular values. Any 
 *                       singular values (and the corresponding column vectors 
 *                       of the U and V matrices) that satisfy 
 *                       :math:`s_i < t * s_0` will be discarded (:math:`s_i` 
 *                       is i-th singular value, :math:`t` is the 
 *                       ``svd_threshold``, and :math:`s_0` is the largest 
 *                       singular value).
 *
 * :param ierr: Error code corresponding to :c:enum:`ErrorCode`. On successful 
 *              completion of the function, :c:enum:`ErrorCode.SUCCESS` is 
 *              returned. Otherwise, a corresponding error code is set.
 *              Must NOT be ``NULL`` pointer - passing in ``NULL`` as ``ierr`` 
 *              is undefined behaviour.
 *
 * :return: The error code from the SVD routine.
 *
 * .. warning::
 *
 *     No clean-up is performed if the compression fails for any 
 *     reason. Therefore, it is likely that the data will be 
 *     partially allocated. If that is the case, it must be 
 *     freed (e.g. using :c:func:`free_tree_data`) before attempting
 *     to use this function again to avoid memory leaks.
 */
int dense_to_tree_hodlr(struct TreeHODLR *restrict hodlr, 
                        const int m,
                        const int *ms,
                        double *restrict matrix, 
                        const double svd_threshold,
                        int *ierr
#ifdef _TEST_HODLR
                        , void *(*malloc)(size_t size),
                        void(*free)(void *ptr)
#endif
                        ) {
  if (hodlr == NULL || matrix == NULL) {
    *ierr = INPUT_ERROR;
    return 0;
  }
  int m_smaller = m / 2;
  int m_larger = m - m_smaller;
  
  *ierr = SUCCESS;

  int result = 0;

  double *s = malloc(2 * m * sizeof(double));
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

  long n_parent_nodes = hodlr->len_work_queue; 

  struct HODLRInternalNode **queue;
  if (ms == NULL) {
    queue = compute_block_sizes_halves(hodlr, m);
  } else {
    queue = compute_block_sizes_custom(hodlr, ms);
    if (hodlr->root->m != m) {
      *ierr = INPUT_ERROR;
      free(s); free(u);
      return 0;
    }
  }

#ifndef _TEST_HODLR
  copy_diagonal_blocks(matrix, m, queue, n_parent_nodes, ierr);
#else
  copy_diagonal_blocks(matrix, m, queue, n_parent_nodes, ierr, malloc);
#endif

  if (*ierr != SUCCESS) {
    free(s); free(u);
    return 0;
  }

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

