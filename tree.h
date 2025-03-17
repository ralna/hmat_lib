enum NodeType {
  DIAGONAL,
  OFFDIAGONAL
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


struct HODLRNode {
  enum NodeType type;
  union HODLRData data;
  struct HODLRNode* children[4];
};



struct TreeHODLR {
  int depth;
  struct HODLRNode *child;
};



struct TreeHODLR dense_to_tree_hodlr(int m, 
                                     int n,
                                     double *matrix,
                                     double svd_threshold,
                                     int depth);

void free_tree_hodlr(struct TreeHODLR *hodlr);
 
