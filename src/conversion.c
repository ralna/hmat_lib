#include <stdlib.h>
#include <math.h>

#include "../include/flat.h"
#include "../include/tree.h"



void tree_hodlr_to_flat_hodlr(struct TreeHODLR *tree,
                              struct FlatHODLR *flat) {
  flat->data = malloc(sizeof(double *) * tree->root->m);
  flat->indices = malloc(sizeof(int) * tree->root->m * (tree->height+1));

  int len_queue = 1;
  int max_depth_n = (int)pow(2, tree->height - 1);

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **temp_pointer;
  queue[0] = tree->root;

  int *sequence = malloc(max_depth_n * sizeof(int));

  for (int i = 0; i < tree->height-1; i++) {
    sequence[0] = 0;

    for (int k = 0; k < queue[0]->children[1].leaf->data.off_diagonal.m; k++) {
      flat->indices[0] = queue[0]->children[1].leaf->data.off_diagonal.s;
    }

    for (int k = 0; k < queue[0]->children[1].leaf->data.off_diagonal.m; k++) {
      flat->indices[0] = queue[0]->children[2].leaf->data.off_diagonal.s;
    }

    for (int j = 1; j < len_queue; j++) {
      for (int k = 0; k < queue[j]->children[1].leaf->data.off_diagonal.m; k++) {
        flat->indices[(tree->height - (j)) + k * tree->height] = 
          queue[j]->children[1].leaf->data.off_diagonal.s;
      }

      for (int k = 0; k < queue[j]->children[1].leaf->data.off_diagonal.m; k++) {
        flat->indices[j + k * tree->height] = 
          queue[j]->children[2].leaf->data.off_diagonal.s;
      }
    }
  }
}
