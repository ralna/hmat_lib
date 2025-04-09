#ifndef TREE_H
#define TREE_H


enum NodeType {
  DIAGONAL,
  OFFDIAGONAL,
  INTERNAL
};


struct NodeDiagonal {
  double *data;
  int m;
};


struct NodeOffDiagonal {
  double *u;
  double *v;
  int m;
  int s;
  int n;
};

union HODLRData {
  struct NodeDiagonal diagonal;
  struct NodeOffDiagonal off_diagonal;
};


struct HODLRLeafNode {
  enum NodeType type;
  union HODLRData data;
  struct HODLRInternalNode *parent;
};


union HODLRNode {
  struct HODLRLeafNode *leaf;
  struct HODLRInternalNode *internal;
};


struct HODLRInternalNode {
  //enum NodeType type;
  union HODLRNode children[4];
  struct HODLRInternalNode *parent;
  int m;
};


struct TreeHODLR {
  int height;
  struct HODLRInternalNode *root;
  struct HODLRLeafNode **innermost_leaves;
};



struct TreeHODLR* allocate_tree(const int height, int *ierr);

int dense_to_tree_hodlr(struct TreeHODLR *hodlr,
                        int m,
                        double *matrix,
                        double svd_threshold,
                        int *ierr);

void free_tree_hodlr(struct TreeHODLR *hodlr);

static void free_partial_tree_hodlr(struct TreeHODLR *hodlr, 
                                    struct HODLRInternalNode **queue, 
                                    struct HODLRInternalNode **next_level);
 
void free_tree_data(struct TreeHODLR *hodlr);

double * multiply_vector(struct TreeHODLR *hodlr, double *vector, double *out);

#endif

