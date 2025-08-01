#include <stdio.h>

#include "../include/tree.h"

#ifdef _TEST_HODLR
#include <criterion/logging.h>
#endif


void log_hodlr_s_symmetric(const struct TreeHODLR *const hodlr) {
  struct HODLRInternalNode **queue = hodlr->work_queue;
  long len_queue = 1, q_next_node_density = hodlr->len_work_queue;
  long q_current_node_density = q_next_node_density;
  int idx = 0;
  queue[0] = hodlr->root;

  for (int level = 1; level < hodlr->height + 1; level++) {
    q_next_node_density /= 2;
    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;

      #ifdef _TEST_HODLR
      cr_log_info("s=%d (level=%d, i=%d)",
             queue[idx]->children[1].leaf->data.off_diagonal.s, level, parent);
      #else
      printf("s=%d (level=%d, i=%d)\n", 
             queue[idx]->children[1].leaf->data.off_diagonal.s, level, parent);
      #endif

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[idx]->children[3].internal;
      queue[idx] = queue[idx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }
}

