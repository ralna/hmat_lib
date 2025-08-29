#include <stdlib.h>

#include "../include/tree.h"
#include "../include/error.h"
#include "../include/blas_wrapper.h"


/**
 * Multiplies a pair of off-diagonal nodes and a vector.
 *
 * Computes matrix-vector multiplication of both off-diagonal children of an 
 * internal node and a vector, and adds the result to the ``out`` vector:
 *
 * * ``out[:m] += parent->children[1] @ vector[m:m+n]``
 * * ``out[m:m+n] += parent->children[2] @ vector[:m]``
 *
 * where ``m`` and ``n`` are the dimensions of the two children.
 *
 * Parameters
 * ----------
 * parent
 *     Pointer to an internal node whose off-diagonal children to multiply.
 *     Must be fully constructed and its off-diagonal children must contain
 *     data. Must not be ``NULL``.
 * vector
 *     Pointer representing an array holding the portion of the vector to be
 *     multiplied. ``parent->m`` elements starting with the first one in 
 *     ``vector`` will be multiplied. Must not be ``NULL`` and must not 
 *     overlap with any of the other pointers.
 * out
 *     Pointer representing an array to which the results are to be saved.
 *     ``parent->m`` elements starting with the first element in ``out`` will
 *     be added to, the array *must* be populated. The values not being set is 
 *     an undefined behaviour.
 *     Must not be ``NULL`` and must not overlap with any of the other 
 *     pointers.
 * workspace
 *     Pointer representing an array that can be used as a workspace. 
 *     Must be a 1D array of at least length N where N is the number of 
 *     columns of the top-right node on ``parent``.
 *     Must not be ``NULL`` and must not overlap with any of the other 
 *     pointers.
 *
 * Returns
 * -------
 * unsigned int
 *     The offset increment, i.e. the number of elements written by this 
 *     function - the next call of this function should pass in 
 *     ``vector + returned_val``.
 */
static inline unsigned int multiply_off_diagonal_vector(
  const struct HODLRInternalNode *restrict const parent,
  const double *restrict const vector,
  double *restrict const out,
  double *restrict const workspace
) {
  const double one = 1.0, zero = 0.0; const int increment = 1;
  const int m = parent->children[1].leaf->data.off_diagonal.m;
  const int n = parent->children[1].leaf->data.off_diagonal.n;
  int s = parent->children[1].leaf->data.off_diagonal.s;

  dgemv_("T", &n, &s, &one, 
          parent->children[1].leaf->data.off_diagonal.v, 
          &n, vector + m, &increment, 
          &zero, workspace, &increment);

  dgemv_("N", &m, &s, &one, 
          parent->children[1].leaf->data.off_diagonal.u, 
          &m, workspace, &increment,
          &one, out, &increment);

  s = parent->children[2].leaf->data.off_diagonal.s;
  dgemv_("T", &m, &s, &one, 
          parent->children[2].leaf->data.off_diagonal.v, 
          &m, vector, &increment, 
          &zero, workspace, &increment);

  dgemv_("N", &n, &s, &one, 
          parent->children[2].leaf->data.off_diagonal.u, 
          &n, workspace, &increment,
          &one, out + m, &increment);

  return m + n;
}


/**
 * Multiplies a HODLR matrix and a vector.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to a HODLR matrix to multiply. This must be a fully constructed
 *     HODLR tree filled with data. Anything else will lead to undefined 
 *     behaviour. If ``NULL``, the function will immediately abort.
 * vector
 *     Pointer representing an array storing the vector to be multiplied. Must
 *     be of length M, where M is the number of rows of ``hodlr``.
 *     Must not overlap with ``out`` and must be occupied with values - either
 *     will lead to undefined behaviour.
 *     If ``NULL``, the function will immediately abort.
 * out
 *     Pointer representing an array to be used for storing the results of the
 *     multiplication. Must be of at least length M, where M is the number of 
 *     rows of ``hodlr``.
 *     Must not overlap with ``vector`` (overlap leads to undefined 
 *     behaviour), but may be both filled with values (which will be 
 *     overwritten) or empty (i.e. just allocated).
 *     If ``NULL``, a new array is allocated.
 * ierr
 *     Pointer to an integer hodling the :c:member:`ErrorCode.SUCCESS` value. 
 *     Used to signal the success or failure of this function. An error status 
 *     code from :c:enum:`ErrorCode` is written into the pointer 
 *     **if an error occurs**. Must not be ``NULL`` - doing so is undefined.
 *
 * Returns
 * -------
 * double * 
 *     The ``out`` array with the results of the matrix-vector multiplication 
 *     stored inside.
 *
 * Errors
 * ------
 * INPUT_ERROR
 *     If ``hodlr`` or ``vector`` is ``NULL`` or if ``vector`` and ``out`` 
 *     point to the same memory location.
 */
double * multiply_vector(const struct TreeHODLR *restrict const hodlr,
                         const double *restrict const vector,
                         double *restrict out,
                         int *restrict const ierr) {
  if (hodlr == NULL || vector == NULL || vector == out) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  if (out == NULL) {
    out = malloc(hodlr->root->m * sizeof(double));
    if (out == NULL) {
      return NULL;
    }
  }
  *ierr = SUCCESS;

  unsigned int offset = 0, idx=0;
  const int increment = 1;
  const double alpha = 1, beta = 0;
  long n_parent_nodes = hodlr->len_work_queue;

  const int m0 = hodlr->root->children[1].leaf->data.off_diagonal.m ;
  const int n0 = hodlr->root->children[1].leaf->data.off_diagonal.m ;
  const int largest_m = (m0 > n0) ? m0 : n0;

  double *workspace = malloc(largest_m * sizeof(double));

  struct HODLRInternalNode **queue = hodlr->work_queue;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;

    for (int _diagonal = 0; _diagonal < 2; _diagonal++) {
      const int m = hodlr->innermost_leaves[idx]->data.diagonal.m;
      dgemv_("N", &m, &m, &alpha, 
             hodlr->innermost_leaves[idx]->data.diagonal.data, 
             &m, vector + offset, &increment, 
             &beta, out + offset, &increment);
      offset += m;
      idx++;
    }
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0; idx = 0;

    for (int parent = 0; parent < n_parent_nodes; parent++) {
      for (int _child = 0; _child < 2; _child++) {
        offset += multiply_off_diagonal_vector(
          queue[idx], vector + offset, out + offset, workspace
        );

        idx++;
      }

      queue[parent] = queue[2 * parent]->parent;
    }
  }

  multiply_off_diagonal_vector(hodlr->root, vector, out, workspace);

  free(workspace);
        
  return out;
}


