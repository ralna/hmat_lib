#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/tree.h"
#include "../include/error.h"


/**
 * Internal function used to free a partially allocated HODLR tree.
 *
 * Used to clean up after unsuccessful allocation of a HODLR tree via 
 * :c:func:`allocate_tree`.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to a partially allocated HODLR. Must not be ``NULL``, but any 
 *     other part may be unallocated.
 * queue
 *     Pointer representing an array of pointers to internal nodes, used to 
 *     loop over the tree. Must not be ``NULL``. Must be of length at least 
 *     :c:member:`TreeHODLR.len_work_queue`. May be empty. Must not overlap 
 *     with ``next_level``.
 * next_level
 *     Pointer representing an array of pointers to internal nodes, used to 
 *     loop over the tree. Must not be ``NULL``. Must be of length at least 
 *     :c:member:`TreeHODLR.len_work_queue`. May be empty. Must not overlap
 *     with ``queue``.
 */
static void free_partial_tree_hodlr(
  struct TreeHODLR *hodlr, 
  struct HODLRInternalNode **queue, 
  struct HODLRInternalNode **next_level
) {
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
}


/**
 * Initialises an off-diagonal leaf node.
 *
 * Given a pointer to a leaf node and a pointer to its parent node, 
 * initialises the leaf node assumings it's an off-diagonal node. Saves the 
 * parent pointer, sets the appropriate enums,  and initialises the
 * pointers to ``NULL`` and numbers to ``0``.
 *
 * Parameters
 * ----------
 * leaf
 *     Pointer to a leaf node to be initialised. Must not be ``NULL``.
 * parent
 *     Pointer to the parent node of ``leaf``. Must not be ``NULL``.
 */
static inline void initialise_leaf_offdiagonal(
  struct HODLRLeafNode *restrict leaf,
  struct HODLRInternalNode *restrict parent
) {
  leaf->type = OFFDIAGONAL;
  leaf->parent = parent;
  leaf->data.off_diagonal.u = NULL;
  leaf->data.off_diagonal.v = NULL;
  leaf->data.off_diagonal.m = 0;
  leaf->data.off_diagonal.s = 0;
  leaf->data.off_diagonal.n = 0;
}


/**
 * Initialises a diagonal leaf node.
 *
 * Given a pointer to a leaf node and a pointer to its parent node, 
 * initialises the leaf node assuming it's a diagonal node. Saves the parent
 * pointer, sets the appropriate enums, and initialises pointers to ``NULL``
 * and numbers to ``0``.
 *
 * Parameters
 * ----------
 * leaf
 *     Pointer to a leaf node to be initialised. Must not be ``NULL``.
 * parent
 *     Pointer to the parent node of ``leaf``. Must not be ``NULL``.
 */
static inline void initialise_leaf_diagonal(
  struct HODLRLeafNode *restrict leaf,
  struct HODLRInternalNode *restrict parent
) {
  leaf->type = DIAGONAL;
  leaf->parent = parent;
  leaf->data.diagonal.data = NULL;
  leaf->data.diagonal.m = 0;
}


/**
 * Initialises an internal node.
 *
 * Given a pointer to an internal node and a pointer to its parent node,
 * initialises the internal node - saves the parent pointer and initialises
 * all numbers to ``0``.
 *
 * Parameters
 * ----------
 * node
 *     Pointer to an internal node to be initialised. Must not be ``NULL``.
 * parent
 *     Pointer to the parent node of ``node``. May be ``NULL`` only if 
 *     ``node`` is the root node.
 */
static inline void initialise_internal(
  struct HODLRInternalNode *restrict node,
  struct HODLRInternalNode *restrict parent
) {
  node->parent = parent;
  node->m = 0;
}


/**
 * Allocates the HODLR structure piecewise.
 *
 * Allocates all the structs composing the HODLR tree by individually 
 * allocating each struct. The pointers are linked together into the tree and
 * the top-level fields on :c:struct:`TreeHODLR` are computed, but all block
 * sizes and data pointers on the children are initialised to ``0``/``NULL``.
 *
 * Parameters
 * ----------
 * height
 *     The height of the HODLR tree to construct, i.e. the number of times the 
 *     matrix will be split. E.g., ``height==1`` splits the matrix once into 
 *     4 blocks, none of which will be split further, giving a HODLR composed 
 *     of a root internal node, holding 4 terminal leaf nodes. 
 *     Must be 1 or greater - smaller values cause early abort, returning 
 *     ``NULL``.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. A status code from :c:enum:`ErrorCode` is written into the
 *     pointer.
 *     Must not be ``NULL`` - passing in ``NULL`` is undefined behaviour.
 *
 * Returns
 * -------
 * struct TreeHODLR *
 *     Pointer to an empty HODLR, if successful, otherwise ``NULL``. 
 *     Any partially allocated memory is automatically freed on failure and 
 *     ``NULL`` is returned.
 *
 * Errors
 * ------
 * INPUT_ERROR
 *     If ``height < 1``.
 * ALLOCATION_FAILURE
 *     If any of the ``malloc`` calls fails.
 *
 * Warnings
 * --------
 * The tree obtained from this function should only be passed into a function 
 * that fills in the data (e.g. :c:func:`dense_to_tree_hodlr`) - it should 
 * not be used anywhere else.
 *
 * See Also
 * --------
 * allocate_tree_monolithic : Allocates nodes in blocks.
 *
 * Notes
 * -----
 * Since this function allocates each ``struct`` in the tree via a separate 
 * call to ``malloc``, the nodes are likely to end up discontinuous, which may
 * hinder performance even though it is unlikely to be a bottleneck. However, 
 * this approach makes changing the height of the tree much simpler since no 
 * ``realloc``\ s are necessary. Therefore, it might be the better allocation 
 * routine if the height is going to be changed a lot.
 */
struct TreeHODLR* allocate_tree(const int height, int *ierr) {
  if (height < 1) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  const long max_depth_n = (long)pow(2, height - 1);

  struct TreeHODLR *hodlr = malloc(sizeof(struct TreeHODLR));
  if (hodlr == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return NULL;
  }

  hodlr->memory_internal_ptr = NULL;
  hodlr->memory_leaf_ptr = NULL;

  hodlr->height = height;
  hodlr->len_work_queue = max_depth_n;

  hodlr->innermost_leaves = 
    malloc(max_depth_n * 2 * sizeof(struct HODLRLeafNode *));
  if (hodlr->innermost_leaves == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr);
    return NULL;
  }

  hodlr->root = malloc(sizeof(struct HODLRInternalNode));
  initialise_internal(hodlr->root, NULL);

  struct HODLRInternalNode **queue = 
    malloc(max_depth_n * sizeof(struct HODLRInternalNode *));
  if (queue == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr->root); free(hodlr);
    return NULL;
  }
  struct HODLRInternalNode **next_level = 
    malloc(max_depth_n * sizeof(struct HODLRInternalNode *));
  if (next_level == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr->root); free(hodlr); free(queue);
    return NULL;
  } 
  struct HODLRInternalNode **temp_pointer = NULL;
  queue[0] = hodlr->root;

  int len_queue = 1;
  for (int _ = 1; _ < height; _++) {
    for (int j = 0; j < len_queue; j++) {
      // WARNING: If the order of these mallocs, changes the change MUST be 
      // reflected in free_partial_tree_hodlr!

      // OFF-DIAGONAL
      for (int leaf = 1; leaf < 3; leaf++) {
        queue[j]->children[leaf].leaf = malloc(sizeof(struct HODLRLeafNode));
        if (queue[j]->children[leaf].leaf == NULL) {
          *ierr = ALLOCATION_FAILURE;
          free_partial_tree_hodlr(hodlr, queue, next_level);
          free(queue); free(next_level);
          return NULL;
        }
        initialise_leaf_offdiagonal(queue[j]->children[leaf].leaf, queue[j]);
      }

      // DIAGONAL (internal)
      for (int internal = 0; internal < 4; internal+=3) {
        queue[j]->children[internal].internal = 
          malloc(sizeof(struct HODLRInternalNode));
        
        if (queue[j]->children[internal].internal == NULL) {
          *ierr = ALLOCATION_FAILURE;
          free_partial_tree_hodlr(hodlr, queue, next_level);
          free(queue); free(next_level);
          return NULL;
        }
        initialise_internal(queue[j]->children[internal].internal, queue[j]);
      }

      next_level[2 * j] = queue[j]->children[0].internal;
      next_level[2 * j + 1] = queue[j]->children[3].internal;
    }

    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;
    
    len_queue *= 2;
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
    }
    hodlr->innermost_leaves[i * 2] = queue[i]->children[0].leaf;
    hodlr->innermost_leaves[i * 2 + 1] = queue[i]->children[3].leaf;
    
    initialise_leaf_diagonal(queue[i]->children[0].leaf, queue[i]);
    initialise_leaf_offdiagonal(queue[i]->children[1].leaf, queue[i]);
    initialise_leaf_offdiagonal(queue[i]->children[2].leaf, queue[i]);
    initialise_leaf_diagonal(queue[i]->children[3].leaf, queue[i]);
  }

  free(next_level);
  
  hodlr->work_queue = queue;

  *ierr = SUCCESS;
  return hodlr;
}


/**
 * Allocates the HODLR structure in blocks.
 *
 * Allocates all the structs composing the HODLR tree by allocating large 
 * chunks of memory and then assigning the pointers appropriately. The 
 * pointers are linked together into the tree and the top-level fields on 
 * :c:struct:`TreeHODLR` are computed, but all block sizes and data pointers 
 * on the children are initialised to ``0``/``NULL``.
 *
 * Parameters
 * ----------
 * height
 *     The height of the HODLR tree to construct, i.e. the number of times the 
 *     matrix will be split. E.g., ``height==1`` splits the matrix once into 
 *     4 blocks, none of which will be split further, giving a HODLR composed 
 *     of a root internal node, holding 4 terminal leaf nodes. 
 *     Must be 1 or greater - smaller values cause early abort, returning 
 *     ``NULL``.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. A status code from :c:enum:`ErrorCode` is written into the
 *     pointer.
 *     Must not be ``NULL`` - passing in ``NULL`` is undefined behaviour.
 *
 * Returns
 * -------
 * struct TreeHODLR *
 *     Pointer to an empty HODLR, if successful, otherwise ``NULL``. 
 *     Any partially allocated memory is automatically freed on failure and 
 *     ``NULL`` is returned.
 *
 * Errors
 * ------
 * INPUT_ERROR
 *     If ``height < 1``.
 * ALLOCATION_FAILURE
 *     If any of the ``malloc`` calls fails.
 *
 * Warnings
 * --------
 * The tree obtained from this function should only be passed into a function 
 * that fills in the data (e.g. :c:func:`dense_to_tree_hodlr`) - it should 
 * not be used anywhere else.
 *
 * See Also
 * --------
 * allocate_tree : Allocates nodes individually.
 * construct_tree : Used for assigning the pointers.
 *
 * Notes
 * -----
 * Since this function allocates all ``struct``\ s of the same type at once,
 * the nodes are always continuous and so likely to be faster to traverse.
 * Unfortunately, this approach makes changing the height of the tree more 
 * difficult since all memory has to be reallocated. Therefore, it might be 
 * better to avoid this routine if the height is going to be changing a lot.
 */
#ifndef _TEST_HODLR
struct TreeHODLR * allocate_tree_monolithic(const int height, int *ierr) {
#else
struct TreeHODLR * allocate_tree_monolithic(const int height, int *ierr,
                                            void *(*malloc)(size_t size),
                                            void(*free)(void *ptr)) {
#endif
  if (height < 1) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  size_t size_internal_nodes, size_leaf_nodes;
  size_t size_work_queue, size_innermost_leaves;
  *ierr = SUCCESS;

  compute_construct_tree_array_sizes(
    height, &size_internal_nodes, &size_leaf_nodes,
    &size_work_queue, &size_innermost_leaves
  );
  
  struct TreeHODLR *hodlr = malloc(sizeof(struct TreeHODLR));
  if (hodlr == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return NULL;
  }
  struct HODLRInternalNode *internal_nodes = malloc(size_internal_nodes);
  if (internal_nodes == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr);
    return NULL;
  }
  struct HODLRLeafNode *leaf_nodes = malloc(size_leaf_nodes);
  if (leaf_nodes == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr); free(internal_nodes);
    return NULL;
  }
  struct HODLRInternalNode **work_queue = malloc(size_work_queue);
  if (work_queue == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr); free(internal_nodes); free(leaf_nodes);
    return NULL;
  }
  struct HODLRLeafNode **innermost_leaves = malloc(size_innermost_leaves);
  if (innermost_leaves == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(hodlr); free(internal_nodes); free(leaf_nodes); free(work_queue);
    return NULL;
  }
  
  construct_tree(height, hodlr, internal_nodes, leaf_nodes, work_queue, 
                 innermost_leaves, ierr);

  if (*ierr != SUCCESS) {
    free(hodlr); free(internal_nodes); free(leaf_nodes); free(work_queue); 
    free(innermost_leaves);
    return NULL;
  }

  return hodlr;
}


/**
 * Computes the memory required for :c:func:`construct_tree`.
 *
 * Used when manually allocating the memory for the HODLR is desirable. 
 *
 * .. important::
 *
 *    Please note that sizes returned by this function are in bytes and so can
 *    and should be used directly, i.e. ``malloc(size_internal_nodes)`` and
 *    NOT ``malloc(size_internal_nodes * sizeof(struct HODLRInternalNode))``.
 *
 * Parameters
 * ----------
 * height
 *     The height of the tree for which the sizes are to be computed. Must be
 *     ``1`` or greater, with the validity of the value not checked. Passing
 *     in values smaller than ``1`` may result in invalid values.
 * size_internal_nodes
 *     Pointer to a single value into which will be written the size in bytes 
 *     required for all the internal nodes. Must not be ``NULL``.
 * size_leaf_nodes
 *     Pointer to a single value into which will be written the size in bytes 
 *     required for all the leaf nodes. Must not be ``NULL``.
 * size_work_queue
 *     Pointer to a single value into which will be written the size in bytes 
 *     required for the :c:member:`TreeHODLR.work_queue` array. Must not be 
 *     ``NULL``.
 * size_innermost_leaves
 *     Pointer to a single value into which will be written the size in bytes 
 *     required for the :c:member:`TreeHODLR.innermost_leaves` array. Must not
 *     be ``NULL``.
 */
void compute_construct_tree_array_sizes(
  const int height,
  size_t *restrict const size_internal_nodes,
  size_t *restrict const size_leaf_nodes,
  size_t *restrict const size_work_queue,
  size_t *restrict const size_innermost_leaves
) {
  const size_t n = (size_t)(pow(2, height - 1));
  *size_work_queue = n * sizeof(struct HODLRInternalNode *);
  *size_internal_nodes = (2 * n - 1) * sizeof(struct HODLRInternalNode);

  *size_innermost_leaves = 2 * n * sizeof(struct HODLRLeafNode *);
  *size_leaf_nodes = (6 * n - 2) * sizeof(struct HODLRLeafNode);
}


/**
 * Constructs the HODLR structure from chunks of memory for all required
 * ``struct``\ s.
 *
 * Given blocks of memory, constructs a HODLR tree by linking pointers 
 * together and setting the top-level fields on :c:struct:`TreeHODLR`. 
 * However, only initialises children's block sizes to ``0`` and data pointers
 * to ``NULL``.
 *
 * The sizes of all the memory chunks can and should be obtained via
 * :c:func:`compute_construct_tree_array_sizes`.
 *
 * Parameters
 * ----------
 * height
 *     The height of the HODLR tree to construct, i.e. the number of times the 
 *     matrix will be split. E.g., ``height==1`` splits the matrix once into 
 *     4 blocks, none of which will be split further, giving a HODLR composed 
 *     of a root internal node, holding 4 terminal leaf nodes. 
 *     Must be 1 or greater - smaller values cause early abort, returning 
 *     ``NULL``.
 * hodlr
 *     Pointer to the HODLR to construct. This must be a single allocated
 *     ``struct`` with no values or pointers set. Must not be ``NULL``.
 * internal_nodes
 *     Pointer representing an array of internal nodes. Each node in the array
 *     will be placed appropriately within the final tree. Must be of length
 *     :math:`2^{height} - 1`. Must not be ``NULL``.
 * leaf_nodes
 *     Pointer representing an array of leaf nodes. Each node in the array 
 *     will be placed appropriately within the final tree. Must be of length
 *     :math:`3 * 2^{height} - 2`. Must not be ``NULL``.
 * work_queue
 *     Pointer representing an array of pointers to internal nodes. This will
 *     be used to set :c:member:`TreeHODLR.work_queue`. Must be of length
 *     :c:mebmber:`TreeHODLR.len_work_queue`. Must not be ``NULL``.
 * innermost_leaves
 *     Pointer representing an array of pointers to leaf nodes. This will be
 *     used to set :c:member:`TreeHODLR.innermost_leaves`. Must be of length
 *     :math:`2^{height}`. Must not be ``NULL``.
 * ierr
 *     Pointer to an integer used to signal the success or failure of this 
 *     function. A status code from :c:enum:`ErrorCode` is written into the
 *     pointer.
 *     Must not be ``NULL`` - passing in ``NULL`` is undefined behaviour.
 */
void construct_tree(const int height,
                    struct TreeHODLR *hodlr,
                    struct HODLRInternalNode *internal_nodes,
                    struct HODLRLeafNode *leaf_nodes,
                    struct HODLRInternalNode **work_queue,
                    struct HODLRLeafNode **innermost_leaves,
                    int *ierr) {
  if (height < 1) {
    *ierr = INPUT_ERROR;
    return;
  }
  *ierr = SUCCESS;

  int len_queue = 1, idx_internal = 1, idx_leaf = 0, qidx = 0;
  const long max_depth_n = (long)pow(2, height - 1);
  long n_parent_nodes = max_depth_n;

  hodlr->memory_internal_ptr = internal_nodes;
  hodlr->memory_leaf_ptr = leaf_nodes;

  hodlr->height = height;
  hodlr->len_work_queue = max_depth_n;
  hodlr->work_queue = work_queue;
  hodlr->innermost_leaves = innermost_leaves;

  hodlr->root = &internal_nodes[0];
  hodlr->root->parent = NULL;
  hodlr->root->m = 0;
  work_queue[0] = hodlr->root;

  for (int _ = 1; _ < hodlr->height; _++) {
    n_parent_nodes /= 2;
    for (int parent = 0; parent < len_queue; parent++) {
      qidx = 2 * parent * n_parent_nodes;

      for (int leaf = 1; leaf < 3; leaf++) {
        work_queue[qidx]->children[leaf].leaf = leaf_nodes + idx_leaf;
        initialise_leaf_offdiagonal(work_queue[qidx]->children[leaf].leaf, 
                                    work_queue[qidx]);
        idx_leaf += 1;
      }

      for (int internal = 0; internal < 4; internal+=3) {
        work_queue[qidx]->children[internal].internal = 
          internal_nodes + idx_internal;
        initialise_internal(work_queue[qidx]->children[internal].internal,
                            work_queue[qidx]);
        idx_internal += 1;
      }

      work_queue[(2 * parent + 1) * n_parent_nodes] = 
        work_queue[qidx]->children[3].internal;
      work_queue[qidx] = work_queue[qidx]->children[0].internal;
    }
    len_queue *= 2;
  }

  for (int i = 0; i < len_queue; i++) {
    for (int leaf = 1; leaf < 3; leaf++) {
      work_queue[i]->children[leaf].leaf = leaf_nodes + idx_leaf;
      initialise_leaf_offdiagonal(work_queue[i]->children[leaf].leaf, 
                                  work_queue[i]);
      idx_leaf += 1;
    }

    for (int leaf = 0; leaf < 4; leaf+=3) {
      work_queue[i]->children[leaf].leaf = leaf_nodes + idx_leaf;
      initialise_leaf_diagonal(work_queue[i]->children[leaf].leaf, 
                               work_queue[i]);
      idx_leaf += 1;
    }
    hodlr->innermost_leaves[i * 2] = work_queue[i]->children[0].leaf;
    hodlr->innermost_leaves[i * 2 + 1] = work_queue[i]->children[3].leaf;
  }
}


/**
 * Frees the allocated data in a HODLR tree.
 *
 * Does NOT free the tree structure, only the data that can be allocated by a 
 * function that fills in the tree, such as :c:func:`dense_to_tree_hodlr`. 
 * 
 * May be used even if the compression function failed, resulting in partial 
 * allocation of data.
 *
 * Additionally, sets all the data pointers to ``NULL``.
 *
 * Parameters
 * ----------
 * hodlr
 *     Pointer to a HODLR tree whose data to free. If ``NULL``, returns 
 *     immediately.
 *
 * See Also
 * --------
 * free_tree_hodlr : Frees both the data and all nodes composing the HODLR.
 */
#ifndef _TEST_HODLR
void free_tree_data(struct TreeHODLR *hodlr) {
#else
void free_tree_data(struct TreeHODLR *hodlr, void(*free)(void *ptr)) {
#endif
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
 * Frees all the allocated data *and* the entire tree structure, including all 
 * the structs etc. I.e. completely cleans up a HODLR tree. Additionally, 
 * sets all the intermediate values as well as the HODLR tree itself to 
 * ``NULL``.
 *
 * .. note::
 *
 *    Please note that this function is agnostic to the way the HODLR was 
 *    allocated (whether individually or by chunks) - it works regardless.
 *
 * Parameters
 * ----------
 * hodlr_ptr
 *     A pointer to a pointer to a pointer to a HODLR tree to free. I.e., must
 *     be a pointer to a dynamically allocated HODLR tree; if ``hodlr_ptr``
 *     is an array of pointers to HODLR trees, only the first tree will be 
 *     freed. If either ``hodlr_ptr`` is ``NULL`` or it points to ``NULL``, 
 *     the function aborts immediately.
 *
 * Warnings
 * --------
 * This function may be unsuitable if the tree was allocated manually in an 
 * unusual way. It can still be used if it was allocated manually in the usual
 * way (allocating 4 separate arrays of sizes given by 
 * :c:func:`compute_construct_tree_array_sizes` and then passing those 4 
 * arrays to :c:func:`construct_tree`), but anything else (such as allocating
 * everything in one big chunk etc.) may result in double frees or other 
 * undefined behaviour and so might require a manual deallocation.
 *
 * See Also
 * --------
 * free_tree_data : Frees only the data - leaves the tree alone.
 */
#ifndef _TEST_HODLR
void free_tree_hodlr(struct TreeHODLR **hodlr_ptr) {
#else
void free_tree_hodlr(struct TreeHODLR **hodlr_ptr,
                     void(*free)(void *ptr)) {
#endif
  if (hodlr_ptr == NULL) {
    return;
  }
  struct TreeHODLR *hodlr = *hodlr_ptr;

  if (hodlr == NULL) {
    return;
  }

  if (hodlr->memory_leaf_ptr != NULL) {
#ifndef _TEST_HODLR
    free_tree_data(hodlr);
#else
    free_tree_data(hodlr, free);
#endif
    free(hodlr->memory_leaf_ptr);
    free(hodlr->memory_internal_ptr);
    free(hodlr->work_queue);
    free(hodlr->innermost_leaves);
    free(hodlr);
    *hodlr_ptr = NULL;
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

