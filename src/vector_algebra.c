#include <stdio.h>
#include <stdlib.h>

#include "../include/tree.h"
#include "../include/error.h"
#include "../include/blas_wrapper.h"


/**
 * Multiplies off-diagonal nodes and a vector
 *
 * Computes matrix-vector multiplication of both off-diagonal HODLR blocks 
 * on a node and a vector, and adds the result to the ``out`` matrix.
 *
 * :param parent: Pointer to an internal node holding the off-diagonal nodes
 *                to multiply.
 * :param vector: Pointer to an array holding the vector being multiplied.
 *                This should be a pointer to the start of the array, and the
 *                ``offset_ptr`` parameter should be used for aligning the 
 *                vector and the HODLR blocks.
 *                Must not overlap with any of the other pointers - doing so
 *                is an undefined behaviour.
 * :param out: Pointer to an array to which the results are to be saved. This
 *             should be a pointer to the start of the array, and the 
 *             ``offset_ptr`` parameter should be used for
 *             aligning it with the HODLR blocks.
 *             This array *must* be populated since the results are added to 
 *             it. The values not being set is an undefined behaviour.
 *             Must not overlap with any of the other pointers - doing so is 
 *             an undefined behaviour.
 * :param workspace: Pointer to an array that can be used as a workspace. 
 *                   Must be of at least length N where N is the number of 
 *                   columns of the top-right node on ``parent``.
 *                   Must not overlap with any of the other arrays - doing 
 *                   so is an undefined behaviour.
 * :param workspace2: Pointer to an array that can be used as a workspace. 
 *                    Must be of at least length M where M is the number of 
 *                    rows of the top-right node on ``parent``.
 *                    Must not overlap with any of the other arrays - doing 
 *                    so is an undefined behaviour.
 * :param alpha: Parameter "alpha" of the BLAS ``dgemv`` routine. Must be 1.
 * :param beta: Parameter "beta" of the BLAS ``dgemv`` routine. Must be 0.
 * :param increment: Constant of 1.
 * :param offset_ptr: Pointer to a single value of offset. This is used as the
 *                    offset into ``vector`` for the top-right node and as 
 *                    the offset into ``out`` for the bottom-left node.
 *                    Must not overlap with any of the other pointers - doing
 *                    so is an undefined behaviour. Similarly, it must not be
 *                    ``NULL`` - again undefined.
 */
static inline void multiply_off_diagonal_vector(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict vector,
  double *restrict out,
  double *restrict workspace,
  double *restrict workspace2,
  const double alpha,
  const double beta,
  const int increment,
  int *restrict offset_ptr
) {
  int i = 0;
  int m = parent->children[1].leaf->data.off_diagonal.m;
  int n = parent->children[1].leaf->data.off_diagonal.n;
  int s = parent->children[1].leaf->data.off_diagonal.s;

  int offset2 = *offset_ptr;
  *offset_ptr += m;
  int offset = *offset_ptr;

  dgemv_("T", &n, &s, &alpha, 
          parent->children[1].leaf->data.off_diagonal.v, 
          &n, vector + offset, &increment, 
          &beta, workspace, &increment);

  dgemv_("N", &m, &s, &alpha, 
          parent->children[1].leaf->data.off_diagonal.u, 
          &m, workspace, &increment,
          &beta, workspace2, &increment);

  for (i = 0; i < m; i++) {
    out[offset2 + i] += workspace2[i];
  }
  
  s = parent->children[2].leaf->data.off_diagonal.s;
  dgemv_("T", &m, &s, &alpha, 
          parent->children[2].leaf->data.off_diagonal.v, 
          &m, vector + offset2, &increment, 
          &beta, workspace, &increment);

  dgemv_("N", &n, &s, &alpha, 
          parent->children[2].leaf->data.off_diagonal.u, 
          &n, workspace, &increment,
          &beta, workspace2, &increment);

  for (i = 0; i < n; i++) {
    out[offset + i] += workspace2[i];
  }
  *offset_ptr += n;
}


/**
 * Multiplies a tree HODLR matrix by a vector.
 *
 * :param hodlr: Pointer to a tree HODLR matrix to multiply. This must be a 
 *               fully constructed HODLR tree, including all the data being 
 *               filled in. If the data has not been assigned (e.g. by using
 *               :c:func:`dense_to_tree_hodlr`), it will lead to undefined
 *               behaviour.
 *               If ``NULL``, the function will immediately abort.
 * :param vector: Pointer to an array storing the vector to use for the
 *                multiplication. Must be of length M, where M is the number
 *                of rows of ``hodlr``.
 *                Must not overlap with ``out`` and must be occupied with 
 *                values - either will lead to undefined behaviour.
 *                If ``NULL``, the function will immediately abort.
 * :param out: Pointer to an array to be used for storing the results of the
 *             multiplication. Must be of at least length M, where M is the 
 *             number of rows of ``hodlr``.
 *             Must not overlap with ``vector``, otherwise undefined behaviour
 *             will ensue, but may be both filled with value (which will be
 *             overwritten) or empty (i.e. just allocated).
 *             If ``NULL``, a new array is allocated.
 * :return: The ``out`` array with the results of the matrix-vector 
 *          multiplication stored inside.
 */
double * multiply_vector(const struct TreeHODLR *restrict hodlr,
                         const double *restrict vector,
                         double *restrict out) {
  if (hodlr == NULL || vector == NULL) {
    return NULL;
  }
  if (out == NULL) {
    out = malloc(hodlr->root->m * sizeof(double));
    if (out == NULL) {
      return NULL;
    }
  }

  int offset = 0, i=0, j=0, k=0, idx=0, m = 0;
  const int increment = 1;
  const double alpha = 1, beta = 0;
  long n_parent_nodes = hodlr->len_work_queue;

  const int largest_m = hodlr->root->children[1].leaf->data.off_diagonal.m ;
  double *workspace = malloc(2 * largest_m * sizeof(double));
  double *workspace2 = workspace + largest_m;

  struct HODLRInternalNode **queue = hodlr->work_queue;

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      idx = 2 * i + j;
      m = hodlr->innermost_leaves[idx]->data.diagonal.m;
      dgemv_("N", &m, &m, &alpha, 
             hodlr->innermost_leaves[idx]->data.diagonal.data, 
             &m, vector + offset, &increment, 
             &beta, out + offset, &increment);
      offset += m;
    }
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        multiply_off_diagonal_vector(
          queue[idx], vector, out, workspace, workspace2, 
          alpha, beta, increment, &offset
        );

        idx += 1;
      }

      queue[j] = queue[2 * j + 1]->parent;
    }
  }

  offset = 0;
  multiply_off_diagonal_vector(
    hodlr->root, vector, out, workspace, workspace2, 
    alpha, beta, increment, &offset
  );

  free(workspace);
        
  return out;
}


