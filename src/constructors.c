#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
 * Computes and sets the sizes of all HODLR internal nodes.
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
 * :param queue: An empty array of pointers to internal nodes. Does not have 
 *               to be empty, but its contents will be overwritten. Must of 
 *               size at least :c:member:`TreeHODLR.len_work_queue`.
 *               Must not be NULL; otherwise is undefined.
 * :param m: The number of rows of the full HODLR matrix.
 */
static void compute_block_sizes(struct TreeHODLR *restrict hodlr,
                                struct HODLRInternalNode **restrict queue,
                                int m) {
  long len_queue = 1, n_parent_nodes = hodlr->len_work_queue;
  int m_smaller = 0, m_larger = 0, idx = 0;
  
  hodlr->root->m = m;
  queue[0] = hodlr->root;

  for (int _ = 1; _ < hodlr->height; _++) {
    n_parent_nodes /= 2;
    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * n_parent_nodes;
      
      m_smaller = queue[idx]->m / 2;
      m_larger = queue[idx]->m - m_smaller;

      queue[idx]->children[0].internal->m = m_larger;
      queue[idx]->children[3].internal->m = m_smaller;

      queue[(2 * parent + 1) * n_parent_nodes] = 
        queue[idx]->children[3].internal;
      queue[2 * idx] = queue[idx]->children[0].internal;
    }
    len_queue = len_queue * 2;
  }
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
                                 int *restrict ierr) {
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
                                 int *restrict ierr) {
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
                           int *ierr) {
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

#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m)
        {
          result = compress_off_diagonal(
            node, 
            m_larger, m_smaller, m_smaller, m, sub_matrix_pointer,
            s + offset_s, u + offset_u, vt + offset_u, svd_threshold, ierr
          ); // make sure ierr is not overwritten

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

#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m)
        {
          result = compress_off_diagonal(
            node, 
            m_smaller, m_larger, m_smaller, m, sub_matrix_pointer, 
            s + offset_s, u + offset_u, vt + offset_u, svd_threshold, ierr
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
#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m)
  {
    result = compress_off_diagonal(
      node, 
      m_larger, m_smaller, m_smaller, m, sub_matrix_pointer,
      s + offset_s, u + offset_u, vt + offset_u, svd_threshold, ierr
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
  // Off-diagonal block in the bottom left corner
  sub_matrix_pointer = matrix + m_larger;
  node = &(queue[0]->children[2].leaf->data.off_diagonal);

#pragma omp task default(none) private(result) firstprivate(node, m_larger, m_smaller, sub_matrix_pointer, offset_s, offset_u) shared(s, u, vt, svd_threshold, ierr, final_result, m)
  {
    result = compress_off_diagonal(
      node, 
      m_smaller, m_larger, m_smaller, m, sub_matrix_pointer, 
      s + offset_s, u + offset_u, vt + offset_u, svd_threshold, ierr
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
                        double *restrict matrix, 
                        const double svd_threshold,
                        int *ierr) {
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
  struct HODLRInternalNode **queue = hodlr->work_queue;

  compute_block_sizes(hodlr, queue, m);

  copy_diagonal_blocks(matrix, m, queue, n_parent_nodes, ierr);
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
        result = compress_matrix(hodlr, queue, matrix, m, s, u, vt, 
                                 svd_threshold, ierr);
      }
    }
  }

  free(s); free(u);
  
  return result;
}


/**
 * Internal function used to free a partially allocated HODLR tree.
 *
 * Used to clean up after unsuccessful allocation of a HODLR tree
 * via :c:func:`allocate_tree`.
 *
 * :param hodlr: Pointer to a HODLR tree.
 * :param queue: A dynamic array of pointers to internal nodes.
 *               Used as a workspace.
 * :param next_level: Dynamic array of pointers to internal nodes.
 *                    Used as a workspace.
 *
 * :return: Nothing
 */
static void free_partial_tree_hodlr(struct TreeHODLR *hodlr, 
                                    struct HODLRInternalNode **queue, 
                                    struct HODLRInternalNode **next_level) {
  int len_queue = 1;
  if (hodlr->root == NULL) {
    return;
  }
  queue[0] = hodlr->root;
  free(hodlr->innermost_leaves);

  struct HODLRInternalNode **temp_pointer = NULL;

  for (int _ = 1; _ < hodlr->height; _++) {
    for (int j = 0; j < len_queue; j++) {
      if (queue[j]->children[1].leaf == NULL) {
        free(queue[j]);
        return;
      }
      free(queue[j]->children[1].leaf);

      if (queue[j]->children[2].leaf == NULL) {
        free(queue[j]);
        return;
      }
      free(queue[j]->children[2].leaf);

      if (queue[j]->children[0].internal == NULL) {
        free(queue[j]);
        return;
      }
      next_level[2 * j] = queue[j]->children[0].internal;
      
      if (queue[j]->children[3].internal == NULL) {
        free(queue[j]);
        return;
      }
      next_level[2 * j + 1] = queue[j]->children[3].internal;
      
      free(queue[j]);
    }
    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;
    
    len_queue = len_queue * 2;
  }

  for (int i = 0; i < len_queue; i++) {
    for (int child = 0; child < 4; child++) {
      if (queue[i]->children[child].leaf == NULL) {
        return;
      }
      free(queue[i]->children[child].leaf);
    }
    free(queue[i]);
  }
  free(hodlr);
  hodlr = NULL;
}


/**
 * Allocates the HODLR tree without allocating the data.
 *
 * Allocates all the structs composing the HODLR tree structure using
 * the ``malloc`` function from ``stdlib``. All the known data is 
 * filled in and pointers are assigned, but no data is allocated or
 * computed. Any such values are not defined are left to the compiler
 * to (potentially) set, so accessing them is undefined behaviour.
 *
 * The tree obtained from this function should only be passed into 
 * a function that fills in the data (e.g. 
 * :c:func:`dense_to_tree_hodlr`) - it should not be used anywhere 
 * else.
 *
 * :param height: The height of the HODLR tree to construct, i.e. the
 *                number of times the matrix will be split. E.g., 
 *                ``height==1`` splits the matrix once into 4 blocks,
 *                none of which will be split further, giving a HODLR
 *                composed of a root internal node, holding 4 terminal
 *                leaf nodes. 
 *                Must be 1 or greater - smaller values cause early 
 *                abort, returning ``NULL``.
 *
 * :param ierr: Error code corresponding to :c:enum:`ErrorCode`. On 
 *              successful completion of the function, 
 *              :c:enum:`ErrorCode.SUCCESS` is returned. Otherwise,
 *              a corresponding error code is set and ``NULL`` is
 *              returned.
 *              Must NOT be ``NULL`` pointer - passing in ``NULL``
 *              as ``ierr`` is undefined behaviour.
 *
 * :return: A pointer to an empty :c:struct:`TreeHODLR` on successful 
 *          allocation, otherwise NULL. Any partially allocated 
 *          memory is automatically freed on failure and NULL is 
 *          returned.
 */
struct TreeHODLR* allocate_tree(const int height, int *ierr) {
  if (height < 1) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  int len_queue = 1;
  const long max_depth_n = (long)pow(2, height - 1);

  struct TreeHODLR *hodlr = malloc(sizeof(struct TreeHODLR));
  if (hodlr == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return NULL;
  }
  //struct HODLRInternalNode *node = (HODLRInternalNode *)malloc(sizeof(HODLRInternalNode));

  hodlr->height = height;
  hodlr->len_work_queue = max_depth_n;

  hodlr->innermost_leaves = malloc(max_depth_n * 2 * sizeof(struct HODLRLeafNode *));
  if (hodlr->innermost_leaves == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr);
    return NULL;
  }

  hodlr->root = malloc(sizeof(struct HODLRInternalNode));
  hodlr->root->parent = NULL;

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(struct HODLRInternalNode *));
  if (queue == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr->root); free(hodlr);
    return NULL;
  }
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(struct HODLRInternalNode *));
  if (next_level == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr->root); free(hodlr); free(queue);
    return NULL;
  } 
  struct HODLRInternalNode **temp_pointer = NULL;
  queue[0] = hodlr->root;

  for (int _ = 1; _ < height; _++) {
    for (int j = 0; j < len_queue; j++) {
      // WARNING: If the order of these mallocs changes the change MUST be reflected
      // in free_partial_tree_hodlr!

      // OFF-DIAGONAL
      for (int leaf = 1; leaf < 3; leaf++) {
        queue[j]->children[leaf].leaf = malloc(sizeof(struct HODLRLeafNode));
        if (queue[j]->children[leaf].leaf == NULL) {
          *ierr = ALLOCATION_FAILURE;
          free_partial_tree_hodlr(hodlr, queue, next_level);
          free(queue); free(next_level);
          return NULL;
        }
        queue[j]->children[leaf].leaf->type = OFFDIAGONAL;
        queue[j]->children[leaf].leaf->parent = queue[j];
      }

      // DIAGONAL (internal)
      for (int leaf = 0; leaf < 4; leaf+=3) {
        queue[j]->children[leaf].internal = malloc(sizeof(struct HODLRInternalNode));
        if (queue[j]->children[leaf].internal == NULL) {
          *ierr = ALLOCATION_FAILURE;
          free_partial_tree_hodlr(hodlr, queue, next_level);
          free(queue); free(next_level);
          return NULL;
        }
        //queue[j]->children[leaf].internal.type = DIAGONAL;
        queue[j]->children[leaf].internal->parent = queue[j];
      }

      next_level[2 * j] = queue[j]->children[0].internal;
      next_level[2 * j + 1] = queue[j]->children[3].internal;
    }

    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;
    
    len_queue = len_queue * 2;
  }

  for (int i = 0; i < len_queue; i++) {
    for (int leaf = 0; leaf < 4; leaf++) {
      queue[i]->children[leaf].leaf = malloc(sizeof(struct HODLRLeafNode));
      if (queue[i]->children[leaf].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(hodlr, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      
      queue[i]->children[leaf].leaf->parent = queue[i];
    }
    hodlr->innermost_leaves[i * 2] = queue[i]->children[0].leaf;
    hodlr->innermost_leaves[i * 2 + 1] = queue[i]->children[3].leaf;
    
    queue[i]->children[0].leaf->type = DIAGONAL;
    queue[i]->children[1].leaf->type = OFFDIAGONAL;
    queue[i]->children[2].leaf->type = OFFDIAGONAL;
    queue[i]->children[3].leaf->type = DIAGONAL;
  }

  free(next_level);
  
  hodlr->work_queue = queue;

  *ierr = SUCCESS;
  return hodlr;
}


/**
 * Frees the allocated data in a HODLR tree.
 *
 * Does NOT free the tree structure, only the data that can
 * be allocated by a function that fills in the tree, such 
 * as :c:func:`dense_to_tree_hodlr`. 
 * 
 * May be used even if the compression function failed, 
 * resulting in partial allocation of data.
 *
 * Additionally, sets all the data pointers to ``NULL``.
 *
 * :param hodlr: Pointer to a HODLR tree whose data to free.
 *               If ``NULL``, returns immediately.
 *
 * :return: Nothing
 */
void free_tree_data(struct TreeHODLR *hodlr) {
  if (hodlr == NULL) {
    return;
  }
  int idx = 0;
  long n_parent_nodes = hodlr->len_work_queue;

  struct HODLRInternalNode **queue = hodlr->work_queue;

  // Loop over nodes one layer up from innermost_leaves
  for (int i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[idx]->parent;

    for (int child = 0; child < 2; child++) {
      free(hodlr->innermost_leaves[idx]->data.diagonal.data);
      hodlr->innermost_leaves[idx]->data.diagonal.data = NULL;
      idx += 1;
    }
  }

  // Loop over the tree (excluding root node)
  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;

    idx = 0;
    for (int j = 0; j < n_parent_nodes; j++) {
      for (int child = 0; child < 2; child ++) {
        for (int leaf = 1; leaf < 3; leaf++) {
          free(queue[idx]->children[leaf].leaf->data.off_diagonal.u);
          free(queue[idx]->children[leaf].leaf->data.off_diagonal.v);
          queue[idx]->children[leaf].leaf->data.off_diagonal.u = NULL;
          queue[idx]->children[leaf].leaf->data.off_diagonal.v = NULL;
        }
        idx += 1;
      }
      queue[j] = queue[idx-1]->parent;
    }
  }

  for (int leaf = 1; leaf < 3; leaf++) {
    free(queue[0]->children[leaf].leaf->data.off_diagonal.u);
    free(queue[0]->children[leaf].leaf->data.off_diagonal.v);
    queue[0]->children[leaf].leaf->data.off_diagonal.u = NULL;
    queue[0]->children[leaf].leaf->data.off_diagonal.v = NULL;
  }
}


/**
 * Frees the entire HOLDR tree.
 *
 * Frees all the allocated data *and* the entire tree structure,
 * including all the structs etc. I.e. completely cleans up a 
 * HODLR tree. Additionally, sets all the intermediate values
 * as well as the HODLR tree itself to ``NULL``.
 *
 * :param hodlr_ptr: A pointer to a pointer to a HODLR tree.
 *                   Must be a pointer to a dynamically 
 *                   allocated HODLR tree; if ``hodlr_ptr``
 *                   is an array of pointers to HODLR trees,
 *                   only the first tree will be freed.
 *                   If either ``hodlr_ptr`` is ``NULL`` or
 *                   it points to ``NULL``, the function 
 *                   aborts immediately.
 *
 * :return: Nothing
 */
void free_tree_hodlr(struct TreeHODLR **hodlr_ptr) {
  if (hodlr_ptr == NULL) {
    return;
  }
  struct TreeHODLR *hodlr = *hodlr_ptr;

  if (hodlr == NULL) {
    return;
  }
  long n_parent_nodes = hodlr->len_work_queue;

  struct HODLRInternalNode **queue = hodlr->work_queue;

  int idx = 0;

  // Loop over nodes one layer up from innermost_leaves
  for (int i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[idx]->parent;

    for (int child = 0; child < 2; child++) {
      free(hodlr->innermost_leaves[idx]->data.diagonal.data);
      free(hodlr->innermost_leaves[idx]);
      hodlr->innermost_leaves[idx] = NULL;

      idx += 1;
    }
  }
  free(hodlr->innermost_leaves);
  hodlr->innermost_leaves = NULL;

  // Loop over the tree (excluding root node)
  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;

    idx = 0;
    for (int j = 0; j < n_parent_nodes; j++) {
      for (int leaf = 1; leaf < 3; leaf++) {
        free(queue[idx]->children[leaf].leaf->data.off_diagonal.u);
        free(queue[idx]->children[leaf].leaf->data.off_diagonal.v);
        free(queue[idx]->children[leaf].leaf);
        queue[idx]->children[leaf].leaf = NULL;
      }

      free(queue[idx]);
      queue[idx] = NULL;
      queue[j] = queue[idx+1]->parent;

      idx += 1;
      for (int leaf = 1; leaf < 3; leaf++) {
        free(queue[idx]->children[leaf].leaf->data.off_diagonal.u);
        free(queue[idx]->children[leaf].leaf->data.off_diagonal.v);
        free(queue[idx]->children[leaf].leaf);
        queue[idx]->children[leaf].leaf = NULL;
      }

      free(queue[idx]);
      queue[idx] = NULL;
      idx += 1;
    }
  }

  for (int leaf = 1; leaf < 3; leaf++) {
    free(queue[0]->children[leaf].leaf->data.off_diagonal.u);
    free(queue[0]->children[leaf].leaf->data.off_diagonal.v);
    free(queue[0]->children[leaf].leaf);
    queue[0]->children[leaf].leaf = NULL;
  }

  free(queue[0]);
  free(hodlr->work_queue); hodlr->work_queue = NULL;
  free(hodlr); *hodlr_ptr = NULL;
}

