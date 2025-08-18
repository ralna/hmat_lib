#include "../include/tree.h"
#include "../include/utils.h"


int get_highest_s(const struct TreeHODLR *const hodlr) {
  long n_parent_nodes = hodlr->len_work_queue;
  struct HODLRInternalNode **queue = hodlr->work_queue;
  int s = 0, highest = 0;

  for (int i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;
  }

  for (int _ = hodlr->height; _ > 0; _--) {
    for (int node = 0; node < n_parent_nodes; node++) {
      for (int leaf = 1; leaf < 3; leaf++) {
        s = queue[node]->children[leaf].leaf->data.off_diagonal.s;
        if (s > highest) highest = s;
      }
      queue[node / 2] = queue[node]->parent;
    }
    n_parent_nodes /= 2;
  }

  return highest;
}
