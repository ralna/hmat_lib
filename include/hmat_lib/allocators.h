#pragma once

#include <stddef.h>

struct TreeHODLR* allocate_tree(const int height, int *ierr);


#ifndef _TEST_HODLR
struct TreeHODLR * allocate_tree_monolithic(const int height, int *ierr);
#else
struct TreeHODLR * allocate_tree_monolithic(const int height, int *ierr,
                                            void *(*malloc)(size_t size),
                                            void(*free)(void *ptr));
#endif


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


void compute_construct_tree_array_sizes(
  const int height,
  size_t *size_internal_nodes,
  size_t *size_leaf_nodes,
  size_t *size_work_queue,
  size_t *size_innermost_leaves
);


void construct_tree(
  const int height,
  struct TreeHODLR *hodlr,
  struct HODLRInternalNode *internal_nodes,
  struct HODLRLeafNode *leaf_nodes,
  struct HODLRInternalNode **work_queue,
  struct HODLRLeafNode **innermost_leaves,
  int *ierr
);
 
