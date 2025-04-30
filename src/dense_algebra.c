#include <stdio.h>
#include <stdlib.h>

#include "../include/lapack_wrapper.h"
#include "../include/tree.h"
#include "../include/error.h"
#include "../include/blas_wrapper.h"


static void print_matrix(const int m, const int n, 
                         const double *matrix, const int lda) {
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      printf("%f    ", matrix[j * lda + i]);
    }
    printf("\n");
  }
  printf("\n");
}


/**
 * Computes the workspace size required for :c:func:`multiply_hodlr_dense`.
 *
 * Given a HODLR tree and the number of columns of the dense matrix, computes 
 * the minimum lengths of the two workspace 
 * arrays used by the :c:func:`multiply_hodlr_dense` function when running 
 * with that HODLR tree.
 *
 * :param hodlr: Pointer to a HODLR tree for which to compute the workspace 
 *               sizes. Must not be ``NULL`` and must be filled with data, 
 *               otherwise is undefined behaviour.
 * :param matrix_n: The number of columns of the dense matrix for which to 
 *                  compute the workspace sizes. Must be greater than 0, 
 *                  other values are undefined behaviour.
 * :param workspace_sizes: Pointer to an array to which the computed workspace
 *                         sizes are saved. The first value will store the 
 *                         size of the first workspace array, and the second
 *                         that of the second workspace array.
 */
void compute_multiply_hodlr_dense_workspace(
  const struct TreeHODLR *restrict hodlr,
  const int matrix_n,
  int *restrict workspace_sizes
) {
  int i = 0, j = 0, k = 0, idx = 0, s = 0;
  long n_parent_nodes = hodlr->len_work_queue;

  struct HODLRInternalNode **queue = hodlr->work_queue;

  workspace_sizes[0] = 1;
  workspace_sizes[1] = hodlr->root->children[1].leaf->data.off_diagonal.m * matrix_n;

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        s = queue[idx + k]->children[1].leaf->data.off_diagonal.s;
        if (s > workspace_sizes[0]) {
          workspace_sizes[0] = s;
        }

        s = queue[idx + k]->children[2].leaf->data.off_diagonal.s;
        if (s > workspace_sizes[0]) {
          workspace_sizes[0] = s;
        }
      }
    } 
  }

  s = hodlr->root->children[1].leaf->data.off_diagonal.s;
  if (s > workspace_sizes[0]) {
    workspace_sizes[0] = s;
  }

  s = hodlr->root->children[2].leaf->data.off_diagonal.s;
  if (s > workspace_sizes[0]) {
    workspace_sizes[0] = s;
  }

  workspace_sizes[0] *= matrix_n;
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
 * :param workspace2: Pointer to an array that can be used as a workspace. 
 *                    Must be of at least size M x N where M is the number of 
 *                    rows of the top-right node on ``parent`` and N is 
 *                    ``matrix_n``.
 *                    Must not overlap with any of the other arrays - doing 
 *                    so is an undefined behaviour.
 * :param alpha: Parameter "alpha" of the BLAS ``dgemv`` routine. Must be 1.
 * :param beta: Parameter "beta" of the BLAS ``dgemv`` routine. Must be 0.
 * :param offset_ptr: Pointer to a single value of offset. This is used as the
 *                    offset into ``vector`` for the top-right node and as 
 *                    the offset into ``out`` for the bottom-left node.
 *                    Must not overlap with any of the other pointers - doing
 *                    so is an undefined behaviour. Similarly, it must not be
 *                    ``NULL`` - again undefined.
 */
static inline void multiply_off_diagonal_dense(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict matrix,
  const int matrix_n,
  const int matrix_ld,
  double *restrict out,
  const int out_ld,
  double *restrict workspace,
  double *restrict workspace2,
  const double alpha,
  const double beta,
  int *restrict offset_ptr
) {
  int i = 0, j = 0;
  int m = parent->children[1].leaf->data.off_diagonal.m;
  int n = parent->children[1].leaf->data.off_diagonal.n;
  int s = parent->children[1].leaf->data.off_diagonal.s;
  
  int offset2 = *offset_ptr;
  *offset_ptr += m;
  int offset = *offset_ptr;

  //print_matrix(m, s, parent->children[1].leaf->data.off_diagonal.u, m);
  dgemm_("T", "N", &s, &matrix_n, &n, &alpha, 
         parent->children[1].leaf->data.off_diagonal.v, 
         &n, matrix + offset, &matrix_ld, 
         &beta, workspace, &s);
  //print_matrix(s, matrix_n, workspace, s);

  dgemm_("N", "N", &m, &matrix_n, &s, &alpha, 
         parent->children[1].leaf->data.off_diagonal.u, 
         &m, workspace, &s,
         &beta, workspace2, &m);
  //print_matrix(m, matrix_n, workspace2, m);

  for (j = 0; j < matrix_n; j++) {
    for (i = 0; i < m; i++) {
      out[offset2 + i + j * out_ld] += workspace2[i + j * m];
    }
  }
  
  s = parent->children[2].leaf->data.off_diagonal.s;
  dgemm_("T", "N", &s, &matrix_n, &m, &alpha, 
         parent->children[2].leaf->data.off_diagonal.v, 
         &m, matrix + offset2, &matrix_ld, 
         &beta, workspace, &s);
  //print_matrix(s, matrix_n, workspace, m);

  dgemm_("N", "N", &n, &matrix_n, &s, &alpha, 
         parent->children[2].leaf->data.off_diagonal.u, 
         &n, workspace, &s,
         &beta, workspace2, &n);
  //print_matrix(n, matrix_n, workspace2, n);

  for (j = 0; j < matrix_n; j++) {
    for (i = 0; i < n; i++) {
      out[offset + i + j * out_ld] += workspace2[i + j * n];
    }
  }
  *offset_ptr += n;
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

  int offset = 0, i=0, j=0, k=0, idx=0, m = 0;
  long n_parent_nodes = hodlr->len_work_queue;
  const double alpha = 1.0, beta = 0.0;

  //print_matrix(hodlr->root->m, matrix_n, matrix, matrix_ld);

  int workspace_size[2] = {0, 0};
  compute_multiply_hodlr_dense_workspace(hodlr, matrix_n, &workspace_size);
  double *workspace = malloc((workspace_size[0] + workspace_size[1]) * sizeof(double));
  double *workspace2 = workspace + workspace_size[0];

  struct HODLRInternalNode **queue = hodlr->work_queue;
  
  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
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

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        multiply_off_diagonal_dense(
          queue[idx], matrix, matrix_n, matrix_ld, 
          out, out_ld, workspace, workspace2, 
          alpha, beta, &offset
        );

        idx += 1;
      }

      queue[j] = queue[2 * j + 1]->parent;
    }
  }

  offset = 0; offset2 = 0;
  multiply_off_diagonal_dense(
    hodlr->root, matrix, matrix_n, matrix_ld, 
    out, out_ld, workspace, workspace2, 
    alpha, beta, &offset
  );

  free(workspace);
        
  return out;
}


