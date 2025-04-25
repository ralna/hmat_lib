#ifndef TREE_STUBS_H
#define TREE_STUBS_H

#include "../../include/tree.h"

int compress_off_diagonal_cr(struct NodeOffDiagonal *, const int, const int, const int, const int, double *, double *, double *, double *, const double, int *);

int dense_to_tree_hodlr_cr(struct TreeHODLR *, const int, double *, const double, int *);

void free_partial_tree_hodlr_cr(struct TreeHODLR *, struct HODLRInternalNode **, struct HODLRInternalNode **);

struct TreeHODLR *allocate_tree_cr(const int, int *);

void free_tree_data_cr(struct TreeHODLR *);

void free_tree_hodlr_cr(struct TreeHODLR **);

#endif
