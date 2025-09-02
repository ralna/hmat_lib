#include <criterion/criterion.h>
#include <criterion/new/assert.h>

#include "./hodlr.h"


void copy_block_sizes(
  const struct TreeHODLR *restrict const src,
  struct TreeHODLR *restrict const dest,
  const bool copy_s
) {
  if (src->height != dest->height) {
    cr_fail("copy_block_sizes: heights do not match (src=%d, dest=%d)",
            src->height, dest->height);
  }

  struct HODLRInternalNode **queue_s = src->work_queue;
  struct HODLRInternalNode **queue_d = dest->work_queue;

  long n_parent_nodes = src->len_work_queue;

  for (int parent = 0; parent < n_parent_nodes; parent++) {
    queue_s[parent] = src->innermost_leaves[2 * parent]->parent;
    queue_d[parent] = dest->innermost_leaves[2 * parent]->parent;

    queue_d[parent]->m = queue_s[parent]->m;

    queue_d[parent]->children[0].leaf->data.diagonal.m =
      queue_s[parent]->children[0].leaf->data.diagonal.m;
    queue_d[parent]->children[3].leaf->data.diagonal.m =
      queue_s[parent]->children[3].leaf->data.diagonal.m;

    queue_d[parent]->children[1].leaf->data.off_diagonal.m =
      queue_s[parent]->children[1].leaf->data.off_diagonal.m;
    queue_d[parent]->children[1].leaf->data.off_diagonal.n =
      queue_s[parent]->children[1].leaf->data.off_diagonal.n;

    queue_d[parent]->children[2].leaf->data.off_diagonal.m =
      queue_s[parent]->children[2].leaf->data.off_diagonal.m;
    queue_d[parent]->children[2].leaf->data.off_diagonal.n =
      queue_s[parent]->children[2].leaf->data.off_diagonal.n;

    if (copy_s == true) {
      queue_d[parent]->children[1].leaf->data.off_diagonal.s =
        queue_s[parent]->children[1].leaf->data.off_diagonal.s;
      queue_d[parent]->children[2].leaf->data.off_diagonal.s =
        queue_s[parent]->children[2].leaf->data.off_diagonal.s;
    }
  }

  for (int _ = src->height - 1; _ > 0; _--) {
    n_parent_nodes /= 2;
    for (int parent = 0; parent < n_parent_nodes; parent++) {
      queue_d[parent] = queue_d[2 * parent]->parent;
      queue_s[parent] = queue_s[2 * parent]->parent;

      queue_d[parent]->m = queue_s[parent]->m;

      queue_d[parent]->children[1].leaf->data.off_diagonal.m =
        queue_s[parent]->children[1].leaf->data.off_diagonal.m;
      queue_d[parent]->children[1].leaf->data.off_diagonal.n =
        queue_s[parent]->children[1].leaf->data.off_diagonal.n;

      queue_d[parent]->children[2].leaf->data.off_diagonal.m =
        queue_s[parent]->children[2].leaf->data.off_diagonal.m;
      queue_d[parent]->children[2].leaf->data.off_diagonal.n =
        queue_s[parent]->children[2].leaf->data.off_diagonal.n;
    }
  }
}


void fill_leaf_node_ints(struct TreeHODLR *hodlr, const int m, int *ss) {
  struct HODLRInternalNode ** queue = hodlr->work_queue;

  long len_queue = 1, q_next_node_density = hodlr->len_work_queue;
  long q_current_node_density = q_next_node_density;
  int m_smaller = m / 2, idx = 0, sidx = 0;
  int m_larger = m - m_smaller;

  hodlr->root->m = m;
  queue[0] = hodlr->root;

  for (int _ = 1; _ < hodlr->height; _++) {
    q_next_node_density /= 2;
    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;
      
      m_smaller = queue[idx]->m / 2;
      m_larger = queue[idx]->m - m_smaller;

      queue[idx]->children[0].internal->m = m_larger;
      queue[idx]->children[3].internal->m = m_smaller;

      queue[idx]->children[1].leaf->data.off_diagonal.m = m_larger;
      queue[idx]->children[1].leaf->data.off_diagonal.s = ss[sidx];
      queue[idx]->children[1].leaf->data.off_diagonal.n = m_smaller;
      sidx++;

      queue[idx]->children[2].leaf->data.off_diagonal.m = m_smaller;
      queue[idx]->children[2].leaf->data.off_diagonal.s = ss[sidx];
      queue[idx]->children[2].leaf->data.off_diagonal.n = m_larger;
      sidx++;
      
      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[idx]->children[3].internal;
      queue[idx] = queue[idx]->children[0].internal;
    }
    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  for (int parent = 0; parent < len_queue; parent++) {
    m_smaller = queue[parent]->m / 2;
    m_larger = queue[parent]->m - m_smaller;

    queue[parent]->children[1].leaf->data.off_diagonal.m = m_larger;
    queue[parent]->children[1].leaf->data.off_diagonal.s = ss[sidx];
    queue[parent]->children[1].leaf->data.off_diagonal.n = m_smaller;
    sidx++;

    queue[parent]->children[2].leaf->data.off_diagonal.m = m_smaller;
    queue[parent]->children[2].leaf->data.off_diagonal.s = ss[sidx];
    queue[parent]->children[2].leaf->data.off_diagonal.n = m_larger;
    sidx++;
  }
}

