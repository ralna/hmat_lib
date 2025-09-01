#include <stddef.h>


#ifndef TREE_H
#define TREE_H

/**
 * The type of of a node.
 */
enum NodeType {
  DIAGONAL,
  OFFDIAGONAL,
  INTERNAL
};


/**
 * Diagonal leaf node
 *
 * This ``struct`` holds the data for the leaf (terminal) node of the HODLR 
 * tree at the diagonal. Therefore, it stores dense data.
 */
struct NodeDiagonal {
  /**
   * The array holding the diagonal data.
   *
   * This is a pointer to an array holding a dense 2D matrix in the 
   * column-major order. The matrix is always a :c:member:`NodeDiagonal.m` x
   * :c:member:`NodeDiagonal.m` square matrix.
   */
  double *data;

  /**
   * The size of the :c:member:`NodeDiagonal.data` matrix.
   */
  int m;
};


/**
 * Off-diagonal leaf node
 *
 * This ``struct`` holds the data for a leaf (terminal) node of the HODLR tree
 * at an off-diagonal position. Therefore, it stores compressed data.
 */
struct NodeOffDiagonal {
  /**
   * The U matrix array
   *
   * This is a pointer to an array holding the U 2D matrix in the column-major
   * order. The matrix is a :c:member:`NodeOffDiagonal.m` x 
   * :c:member:`NodeOffDiagonal.s` matrix. All the values stored are already
   * multiplied by the corresponding singular values and the column vectors are
   * stored in the (descending) order of singular value size.
   */
  double *u;

  /**
   * The V matrix array
   *
   * This is a pointer to an array holding the V 2D matrix in column-major 
   * order. It is a :c:member:`NodeOffDiagonal.n` x 
   * :c:member:`NodeOffDiagonal.s` matrix, with the column vectors stored in 
   * the (descending) order of singular value size.
   *
   * .. important::
   * 
   *     This matrix is stored as V, **not** as :math:`V^T`.
   */
  double *v;

  /**
   * The number of rows of the U matrix
   *
   * This also corresponds to the number of rows of the original off-diagonal 
   * block.
   */
  int m;

  /**
  * The number of singular values 
  *
  * This is the number of singular values kept during the compression and 
  * corresponds to the number of columns of U as well as the number of columns 
  * V.
  */
  int s;

  /**
   * The number of rows of the V matrix
   *
   * This also corresponds to the number of columns of the original 
   * off-diagonal block.
   */
  int n;
};


/**
 * Union encompassing the different types of leaf nodes.
 *
 * It is used inside :c:struct:`HODLRLeafNode` to store either the data for 
 * a diagonal or off-diagonal HODLR block.
 *
 * Please note that this is a union of struct and not of pointers to structs.
 */
union HODLRData {
  /**
   * The data for a diagonal HODLR block.
   */
  struct NodeDiagonal diagonal;

  /** The data for an off-diagonal HODLR block */
  struct NodeOffDiagonal off_diagonal;
};


/**
 * Leaf node of the tree HODLR matrix.
 *
 * Represents a leaf (terminal) node in the tree HODLR structure, containing
 * concrete data, either in the diagonal form or the off-diagonal.
 */
struct HODLRLeafNode {
  /** The node type */
  enum NodeType type;

  /**
   * The data associated with this leaf node.
   *
   * If the node is diagonal, this data is in the form of a diagonal block, and
   * if it is off-diagonal, it is in the off-diagonal form.
   */
  union HODLRData data;

  /** A pointer to the parent node. */
  struct HODLRInternalNode *parent;
};


/**
 * Union encompassing the different types of nodes in the HODLR tree.
 *
 * It represents either a leaf node, which stores data, or an internal node,
 * which contains further children nodes.
 *
 * Please note that this is a union of pointers to structs and not to structs
 * themselves.
 */
union HODLRNode {
  /** Pointer to a leaf node. */
  struct HODLRLeafNode *leaf;

  /** Pointer to an internal node. */
  struct HODLRInternalNode *internal;
};


/**
 * Internal node of the tree HODLR matrix.
 *
 * Represents an internal (non-terminal) node in the tree HODLR structure, 
 * containing pointers to its children nodes and no data.
 */
struct HODLRInternalNode {
  //enum NodeType type;
  
  /**
   * Array of pointers to the children nodes of this internal node.
   *
   * All 4 children are stored in this array in the following order:
   *
   *  1. The top left diagonal block. This is either 
   *
   *     a. :c:member:`HODLRNode.internal` if the children are internal, or
   *     b. :c:member:`HODLRNode.leaf` if the children are terminal, in which 
   *        case the stored data is diagonal and so the 
   *        :c:member:`HODLRLeafNode.data` is :c:member:`HODLRData.diagonal`,
   *        storing the :c:struct:`NodeDiagonal`.
   *
   *  2. The top right block, which is *always* :c:member:`HODLRNode.leaf` and
   *     stores off-diagonal data via :c:member:`HODLRLeafNode.data` being 
   *     :c:member:`HODLRData.off_diagonal`, storing the 
   *     :c:struct:`NodeOffDiagonal`.
   *
   *  3. The bottom left block, which is *always* :c:member:`HODLRNode.leaf` and
   *     stores off-diagonal data via :c:member:`HODLRLeafNode.data` being 
   *     :c:member:`HODLRData.off_diagonal`, storing the 
   *     :c:struct:`NodeOffDiagonal`.
   *
   *  4. The bottom right diagonal block. This is either 
   *
   *     a. :c:member:`HODLRNode.internal` if the children are internal, or
   *     b. :c:member:`HODLRNode.leaf` if the children are terminal, in which 
   *        case the stored data is diagonal and so the 
   *        :c:member:`HODLRLeafNode.data` is :c:member:`HODLRData.diagonal`,
   *        storing the :c:struct:`NodeDiagonal`.
   */
  union HODLRNode children[4];

  /** Pointer to the parent node. NULL if this node is root. */
  struct HODLRInternalNode *parent;

  /** The size of the matrix block represented by this node. */
  int m;
};


/**
 * Tree HODLR matrix.
 *
 * Stores a HODLR matrix in a tree structure.
 */
struct TreeHODLR {
  /** 
   * The height of the HODLR tree. 
   *
   * This is the number of times the original matrix was split, and is equal to
   * the number of layers of internal nodes in the tree. Therefore, a height=1
   * tree will have a root node holding 4 leaf nodes, and a height=2 tree will
   * have a root node, one layer of internal nodes, and then a layer of leaf 
   * nodes.
   */
  int height;

  /** Pointer to the root internal node. */
  struct HODLRInternalNode *root;

  /**
   * Array of pointers to the diagonal leaf nodes.
   *
   * This is a dynamic array that holds a pointer to each 
   * :c:struct:`HODLRLeafNode` with :c:member:`HODLRLeafNode.data` == 
   * :c:member:`HODLRData.diagonal`, all of which are found at the bottom-most
   * layer of the tree. The pointers are ordered from the block at the top left
   * of the matrix to the one at the bottom right. The length of this array is
   * :math:`2^{h}` where ``h`` is :c:member:`TreeHODLR.height` which is also 
   * equal to :c:member:`TreeHODLR.len_work_queue` * 2.
   */
  struct HODLRLeafNode **innermost_leaves;

  /**
   * The length of the :c:member:`TreeHODLR.work_queue` array.
   *
   * Corresponds to the number of internal nodes at the second-to-last layer 
   * of the tree (i.e. the last non-terminal layer). Equals :math:`2^{h-1}`, 
   * where ``h`` is :c:member:`TreeHODLR.height`.
   */
  long len_work_queue;

  /**
   * Internal work array.
   *
   * Dynamic array used internally to loop over the HOLDR tree. It is an array
   * of pointers to internal nodes.
   *
   * .. important::
   *
   *     This array may change even when this ``struct`` is passed as 
   *     ``const``, since it is used as an internal workspace.
   */
  struct HODLRInternalNode **work_queue;

  struct HODLRInternalNode *memory_internal_ptr;
  struct HODLRLeafNode *memory_leaf_ptr;
};


struct TreeHODLR* allocate_tree(const int height, int *ierr);

int dense_to_tree_hodlr(struct TreeHODLR *hodlr,
                        const int m,
                        const int *ms,
                        double *matrix,
                        const double svd_threshold,
                        int *ierr
#ifdef _TEST_HODLR
                        , void *(*permanent_allocator)(size_t size),
                        void(*free)(void *ptr)
#endif
                        );

#ifndef _TEST_HODLR
void free_tree_hodlr(struct TreeHODLR **hodlr_ptr);
#else
void free_tree_hodlr(struct TreeHODLR **hodlr_ptr,
                     void(*free)(void *ptr));
#endif

#ifndef _TEST_HODLR
void free_tree_data(struct TreeHODLR *hodlr);
#else
void free_tree_data(struct TreeHODLR *hodlr, void(*free)(void *ptr));
#endif

double * multiply_vector(
  const struct TreeHODLR *hodlr, 
  const double *vector, 
  double *out,
  int *ierr
);

int multiply_hodlr_hodlr(
  const struct TreeHODLR *hodlr1,
  const struct TreeHODLR *hodlr2,
  struct TreeHODLR *out,
  const double svd_threshold,
  int *ierr
);

int compute_multiply_hodlr_dense_workspace(
  const struct TreeHODLR *hodlr,
  const int matrix_a
);

double * multiply_hodlr_dense(const struct TreeHODLR *hodlr,
                              const double *matrix,
                              const int matrix_n,
                              const int matrix_ld,
                              double *out,
                              const int out_ld,
                              int *ierr);

double * multiply_hodlr_transpose_dense(const struct TreeHODLR *hodlr,
                                        const double *matrix,
                                        const int matrix_n,
                                        const int matrix_ld,
                                        double *out,
                                        const int out_ld,
                                        int *ierr);

void multiply_internal_node_dense(
  const struct HODLRInternalNode *internal,
  const int height,
  const double *matrix,
  const int matrix_n,
  const int matrix_ld,
  const struct HODLRInternalNode **queue,
  double *workspace,
  double *out,
  const int out_ld
);

void multiply_internal_node_transpose_dense(
  const struct HODLRInternalNode *internal,
  const int height,
  const double *matrix,
  const int matrix_n,
  const int matrix_ld,
  const struct HODLRInternalNode **queue,
  double *workspace,
  double *out,
  const int out_ld
);

double * multiply_dense_hodlr(const struct TreeHODLR *hodlr,
                              const double * matrix,
                              const int matrix_m,
                              const int matrix_ld,
                              double * out,
                              const int out_ld,
                              int *ierr);
 
void compute_construct_tree_array_sizes(const int height,
                                        size_t *size_internal_nodes,
                                        size_t *size_leaf_nodes,
                                        size_t *size_work_queue,
                                        size_t *size_innermost_leaves);

void construct_tree(const int height,
                    struct TreeHODLR *hodlr,
                    struct HODLRInternalNode *internal_nodes,
                    struct HODLRLeafNode *leaf_nodes,
                    struct HODLRInternalNode **work_queue,
                    struct HODLRLeafNode **innermost_leaves,
                    int *ierr);
 
#ifndef _TEST_HODLR
struct TreeHODLR * allocate_tree_monolithic(const int height, int *ierr);
#else
struct TreeHODLR * allocate_tree_monolithic(const int height, int *ierr,
                                            void *(*malloc)(size_t size),
                                            void(*free)(void *ptr));
#endif

#endif

