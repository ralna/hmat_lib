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

  *ierr = SUCCESS;
  return result;
}


/**
 * Compresses a dense matrix into a HODLR tree matrix.
 *
 * Requires a preallocated HODLR tree (e.g. from 
 * :c:func:`allocate_tree`), a matrix, and some settings.
 *
 * :param hodlr: A pointer to a HODLR tree. This *should*
 *               be an empty tree, with no data filled in,
 *               since any pointers to data will be 
 *               overwritten with newly allocated data and
 *               therefore lost, potentially causing memory 
 *               leaks.
 *               If ``NULL``, aborts immediately.
 *               Passing in a partially or incorrectly 
 *               allocated tree is an undefined behaviour.
 *
 * :param m: The size of the ``matrix`` matrix. I.e., the
 *           number of rows and columns.
 *
 * :param matrix: The dense matrix to compress. This must 
 *                be an column-major array holding an 
 *                ``m`` x ``m`` square 2D matrix.
 *                If ``NULL``, aborts immediately.
 *                *Some of the values may be overwritten
 *                during SVD compuration*
 *
 * :param svd_threshold: The threshold for discarding 
 *                       singular values. Any singular 
 *                       values (and the corresponding 
 *                       column vectors of the U and V
 *                       matrices) that satisfy 
 *                       :math:`s_i < t * s_0` will be 
 *                       discarded (:math:`s_i` is i-th 
 *                       singular value, :math:`t` is the 
 *                       ``svd_threshold``, and :math:`s_0` 
 *                       is the largest singular value).
 *
 * :param ierr: Error code corresponding to :c:enum:`ErrorCode`. On 
 *              successful completion of the function, 
 *              :c:enum:`ErrorCode.SUCCESS` is returned. Otherwise,
 *              a corresponding error code is set.
 *              Must NOT be ``NULL`` pointer - passing in ``NULL``
 *              as ``ierr`` is undefined behaviour.
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

  // TODO: OPNEMP
  int result = 0, offset = 0, n_singular_values=m_smaller, len_queue=1;
  int idx = 0;
  double *sub_matrix_pointer = NULL; double *data = NULL;
  // TODO: Standardise i and j indices
  hodlr->root->m = m;

  double *s = malloc(n_singular_values * sizeof(double));
  if (s == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return 0;
  }
  double *u = malloc(m_larger * n_singular_values * sizeof(double));
  if (u == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(s);
    return 0;
  }

  double *vt = malloc(n_singular_values * m_larger * sizeof(double));
  if (vt == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(s); free(u);
    return 0;
  }

  long n_parent_nodes = hodlr->len_work_queue; 
  struct HODLRInternalNode **queue = hodlr->work_queue;

  queue[0] = hodlr->root;
  for (int _ = 1; _ < hodlr->height; _++) {
    for (int parent = 0; parent < len_queue; parent++) {
      m_smaller = queue[parent]->m / 2;
      m_larger = queue[parent]->m - m_smaller;

      idx = parent * n_parent_nodes;
      queue[idx]->children[0].internal->m = m_larger;
      queue[idx]->children[3].internal->m = m_smaller;

      queue[2 * idx] = queue[idx]->children[0].internal;
      queue[(2 * parent + 1) * n_parent_nodes] = 
        queue[idx]->children[3].internal;
    }
    len_queue = len_queue * 2;
    n_parent_nodes /= 2;
  }

  n_parent_nodes = hodlr->len_work_queue;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    //queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;
    m_smaller = queue[parent]->m / 2;
    m_larger = queue[parent]->m - m_smaller;

    data = malloc(m_larger * m_larger * sizeof(double));
    if (data == NULL) {
      *ierr = ALLOCATION_FAILURE;
      free(s); free(u); free(vt);
      return 0;
    }
    for (int j = 0; j < m_larger; j++) {
      for (int k = 0; k < m_larger; k++) {
        data[k + j * m_larger] = matrix[k + offset + (j + offset) * m];
      }
    }
    queue[parent]->children[0].leaf->data.diagonal.data = data;
    queue[parent]->children[0].leaf->data.diagonal.m = m_larger;

    offset += m_larger;

    data = malloc(m_smaller * m_smaller * sizeof(double));
    if (data == NULL) {
      *ierr = ALLOCATION_FAILURE;
      free(s); free(u); free(vt);
      return 0;
    }
    for (int j = 0; j < m_smaller; j++) {
      for (int k = 0; k < m_smaller; k++) {
        data[k + j * m_smaller] = matrix[k + offset + (j + offset) * m];
      }
    }
    queue[parent]->children[3].leaf->data.diagonal.data = data;
    queue[parent]->children[3].leaf->data.diagonal.m = m_smaller;

    offset += m_smaller;
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0;
  
    for (int parent = 0; parent < n_parent_nodes; parent++) {
      for (int child = 2 * parent; child < 2 * parent + 2; child++) {
        m_smaller = queue[child]->m / 2;
        m_larger = queue[child]->m - m_smaller;

        sub_matrix_pointer = matrix + offset + m * (offset + m_larger);
        result = compress_off_diagonal(
          &(queue[child]->children[1].leaf->data.off_diagonal), 
          m_larger, m_smaller, m_smaller, m, 
          sub_matrix_pointer,
          s, u, vt, svd_threshold, ierr
        );
        if (*ierr != SUCCESS) {
          //handle_error(ierr, result);  // 
          // error out
          free(s); free(u); free(vt);
          return result;
        }
        
        // Off-diagonal block in the bottom left corner
        sub_matrix_pointer = matrix + m * offset + offset + m_larger;
        result = compress_off_diagonal(
          &(queue[child]->children[2].leaf->data.off_diagonal), 
          m_smaller, m_larger, m_smaller, m,
          sub_matrix_pointer, 
          s, u, vt, svd_threshold, ierr
        );
        if (*ierr != SUCCESS) {
          // error out
          free(s); free(u); free(vt);
          return result;
        }
      }

      queue[parent] = queue[2 * parent + 1]->parent;
    }
  }

  m_smaller = queue[0]->m / 2;
  m_larger = queue[0]->m - m_smaller;

  sub_matrix_pointer = matrix + m * m_larger;
  result = compress_off_diagonal(
    &(queue[0]->children[1].leaf->data.off_diagonal), 
    m_larger, m_smaller, m_smaller, m, 
    sub_matrix_pointer,
    s, u, vt, svd_threshold, ierr
  );
  if (*ierr != SUCCESS) {
    //handle_error(ierr, result);  // 
    // error out
    free(s); free(u); free(vt);
    return result;
  }
  
  // Off-diagonal block in the bottom left corner
  sub_matrix_pointer = matrix + m_larger;
  result = compress_off_diagonal(
    &(queue[0]->children[2].leaf->data.off_diagonal), 
    m_smaller, m_larger, m_smaller, m,
    sub_matrix_pointer, 
    s, u, vt, svd_threshold, ierr
  );
  if (*ierr != SUCCESS) {
    // error out
    free(s); free(u); free(vt);
    return result;
  }

  free(s); free(u); free(vt); 

  *ierr = SUCCESS;
  
  return 0;
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

  for (int i = 1; i < hodlr->height; i++) {
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
    for (int j = 0; j < 4; j++) {
      if (queue[i]->children[j].leaf == NULL) {
        return;
      }
      free(queue[i]->children[j].leaf);
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

  struct TreeHODLR *root = malloc(sizeof(struct TreeHODLR));
  if (root == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return NULL;
  }
  //struct HODLRInternalNode *node = (HODLRInternalNode *)malloc(sizeof(HODLRInternalNode));

  root->height = height;
  root->len_work_queue = max_depth_n;

  root->innermost_leaves = malloc(max_depth_n * 2 * sizeof(struct HODLRLeafNode *));
  if (root->innermost_leaves == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(root);
    return NULL;
  }

  root->root = malloc(sizeof(struct HODLRInternalNode));
  root->root->parent = NULL;

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(struct HODLRInternalNode *));
  if (queue == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(root->root); free(root);
    return NULL;
  }
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(struct HODLRInternalNode *));
  if (next_level == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(root->root); free(root); free(queue);
    return NULL;
  } 
  struct HODLRInternalNode **temp_pointer = NULL;
  queue[0] = root->root;

  for (int i = 1; i < height; i++) {
    for (int j = 0; j < len_queue; j++) {
      // WARNING: If the order of these mallocs changes the change MUST be reflected
      // in free_partial_tree_hodlr!
      queue[j]->children[1].leaf = malloc(sizeof(struct HODLRLeafNode));
      if (queue[j]->children[1].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      queue[j]->children[1].leaf->type = OFFDIAGONAL;
      queue[j]->children[1].leaf->parent = queue[j];

      queue[j]->children[2].leaf = malloc(sizeof(struct HODLRLeafNode));
      if (queue[j]->children[2].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      queue[j]->children[2].leaf->type = OFFDIAGONAL;
      queue[j]->children[2].leaf->parent = queue[j];

      queue[j]->children[0].internal = malloc(sizeof(struct HODLRInternalNode));
      if (queue[j]->children[0].internal == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      //queue[j]->children[0].internal.type = DIAGONAL;
      queue[j]->children[0].internal->parent = queue[j];

      queue[j]->children[3].internal = malloc(sizeof(struct HODLRInternalNode));
      if (queue[j]->children[3].internal == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      //queue[j]->children[3].internal.type = DIAGONAL;
      queue[j]->children[3].internal->parent = queue[j];

      next_level[2 * j] = queue[j]->children[0].internal;
      next_level[2 * j + 1] = queue[j]->children[3].internal;
    }

    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;
    
    len_queue = len_queue * 2;
  }

  for (int i = 0; i < len_queue; i++) {
    for (int j = 0; j < 4; j++) {
      queue[i]->children[j].leaf = malloc(sizeof(struct HODLRLeafNode));
      if (queue[i]->children[j].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      
      queue[i]->children[j].leaf->parent = queue[i];
    }
    root->innermost_leaves[i * 2] = queue[i]->children[0].leaf;
    root->innermost_leaves[i * 2 + 1] = queue[i]->children[3].leaf;
    
    queue[i]->children[0].leaf->type = DIAGONAL;
    queue[i]->children[1].leaf->type = OFFDIAGONAL;
    queue[i]->children[2].leaf->type = OFFDIAGONAL;
    queue[i]->children[3].leaf->type = DIAGONAL;
  }

  free(next_level);
  
  root->work_queue = queue;

  *ierr = SUCCESS;
  return root;
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
  int i = 0, j = 0, k = 0, idx = 0;
  long n_parent_nodes = hodlr->len_work_queue;

  struct HODLRInternalNode **queue = hodlr->work_queue;

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      free(hodlr->innermost_leaves[2 * i + j]->data.diagonal.data);
      hodlr->innermost_leaves[2 * i + j]->data.diagonal.data = NULL;
    }
  }

  for (i = hodlr->height-1; i > 0; i--) {
    n_parent_nodes /= 2;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        idx += k;
        free(queue[idx]->children[1].leaf->data.off_diagonal.u);
        free(queue[idx]->children[1].leaf->data.off_diagonal.v);
        queue[idx]->children[1].leaf->data.off_diagonal.u = NULL;
        queue[idx]->children[1].leaf->data.off_diagonal.v = NULL;
        
        free(queue[idx]->children[2].leaf->data.off_diagonal.u);
        free(queue[idx]->children[2].leaf->data.off_diagonal.v);
        queue[idx]->children[2].leaf->data.off_diagonal.u = NULL;
        queue[idx]->children[2].leaf->data.off_diagonal.v = NULL;
      }

      queue[j] = queue[idx+1]->parent;
    }
  }

  for (i = 1; i < 3; i++) {
    free(queue[0]->children[i].leaf->data.off_diagonal.u);
    free(queue[0]->children[i].leaf->data.off_diagonal.v);
    queue[0]->children[i].leaf->data.off_diagonal.u = NULL;
    queue[0]->children[i].leaf->data.off_diagonal.v = NULL;
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

  int i = 0, j = 0, k = 0, idx = 0;
  if (hodlr == NULL) {
    return;
  }
  long n_parent_nodes = hodlr->len_work_queue;

  struct HODLRInternalNode **queue = hodlr->work_queue;

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      free(hodlr->innermost_leaves[2 * i + j]->data.diagonal.data);
      free(hodlr->innermost_leaves[2 * i + j]);
      hodlr->innermost_leaves[2 * i + j] = NULL;
    }
  }
  free(hodlr->innermost_leaves);
  hodlr->innermost_leaves = NULL;

  for (i = hodlr->height-1; i > 0; i--) {
    n_parent_nodes /= 2;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        free(queue[idx + k]->children[1].leaf->data.off_diagonal.u);
        free(queue[idx + k]->children[1].leaf->data.off_diagonal.v);
        free(queue[idx + k]->children[1].leaf);
        queue[idx + k]->children[1].leaf = NULL;

        free(queue[idx + k]->children[2].leaf->data.off_diagonal.u);
        free(queue[idx + k]->children[2].leaf->data.off_diagonal.v);
        free(queue[idx + k]->children[2].leaf);
        queue[idx + k]->children[1].leaf = NULL;
      }

      free(queue[idx]);
      queue[idx] = NULL;
      queue[j] = queue[idx+1]->parent;
      free(queue[idx+1]);
      queue[idx+1] = NULL;
    }
  }

  for (i = 1; i < 3; i++) {
    free(queue[0]->children[i].leaf->data.off_diagonal.u);
    free(queue[0]->children[i].leaf->data.off_diagonal.v);
    free(queue[0]->children[i].leaf);
    queue[0]->children[i].leaf = NULL;
  }

  free(queue[0]);
  free(hodlr->work_queue); hodlr->work_queue = NULL;
  free(hodlr); *hodlr_ptr = NULL;
}

