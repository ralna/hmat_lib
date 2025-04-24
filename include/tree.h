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
                        const int m,
                        double *matrix,
                        const double svd_threshold,
                        int *ierr);

void free_tree_hodlr(struct TreeHODLR **hodlr_ptr);

void free_tree_data(struct TreeHODLR *hodlr);

double * multiply_vector(const struct TreeHODLR *hodlr, const double *vector, double *out);

double * multiply_hodlr_dense(const struct TreeHODLR *hodlr,
                              const double *matrix,
                              const int matrix_n,
                              const int matrix_ld,
                              double *out,
                              const int out_ld);
#endif

