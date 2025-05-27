#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/tree.h"
#include "../include/error.h"


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
 * Initialises an off-diagonal leaf node.
 *
 * Given a pointer to a leaf node and a pointer to its parent node, 
 * initialises the leaf node assumings it's an off-diagonal node. Saves the 
 * parent pointer, sets the appropriate enums,  and initialises the
 * pointers to ``NULL`` and numbers to ``0``.
 *
 * :param leaf: A pointer to a leaf node to be initialised. ``NULL`` is 
 *              undefined.
 * :param parent: A pointer to ``leaf``'s parent node. Must not be ``NULL``.
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
 * :param leaf: A pointer to a leaf node to be initialised. ``NULL`` is 
 *              undefined.
 * :param parent: A pointer to ``leaf``'s parent node. Must not be ``NULL``.
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
 * :param node: A pointer to an internal node to be initialised. ``NULL`` is
 *              undefined.
 * :param parent: A pointer to the ``node``'s parent node. May be ``NULL`` 
 *                only if ``node`` is the root node.
 */
static inline void initialise_internal(
  struct HODLRInternalNode *restrict node,
  struct HODLRInternalNode *restrict parent
) {
  node->parent = parent;
  node->m = 0;
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

  hodlr->memory_internal_ptr = NULL;
  hodlr->memory_leaf_ptr = NULL;
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
  initialise_internal(hodlr->root, NULL);

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


struct TreeHODLR * allocate_tree_monolithic(const int height, int *ierr) {
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


void compute_construct_tree_array_sizes(const int height,
                                        size_t *size_internal_nodes,
                                        size_t *size_leaf_nodes,
                                        size_t *size_work_queue,
                                        size_t *size_innermost_leaves) {
  const size_t n = (size_t)(pow(2, height - 1));
  *size_work_queue = n * sizeof(struct HODLRInternalNode *);
  *size_internal_nodes = (2 * n - 1) * sizeof(struct HODLRInternalNode);

  *size_innermost_leaves = 2 * n * sizeof(struct HODLRLeafNode *);
  *size_leaf_nodes = (6 * n - 2) * sizeof(struct HODLRLeafNode);
}


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

  if (hodlr->memory_leaf_ptr != NULL) {
    free_tree_data(hodlr);
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

