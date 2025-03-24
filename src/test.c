#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/tree.h"


static void print_matrix(int m, int n, double *matrix) {
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      printf("%f    ", matrix[j * m + i]);
    }
    printf("\n");
  }
  printf("\n");
}


static void print_node_diagonal(struct NodeDiagonal *node) {
  int m = node->m;

  printf("(%dx%d) node=%p data=%p\n", m, m, node, node->data);

  for (int i=0; i<m; i++) {
    for (int j=0; j < m; j++) {
      printf("%f    ", node->data[j * m + i]);
    }
    printf("\n");
  }
  printf("\n");
}


static void print_node_offdiagonal(struct NodeOffDiagonal *node) {
  int m = node->m; int s = node->s; int n = node->n;

  printf("U (%dx%d):\n", m, s);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < s; j++) {
      printf("%f    ", node->u[j * m + i]);
    }
    printf("\n");
  }

  printf("\nV_T (%dx%d):\n", s, n);
  for (int i = 0; i < s; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f    ", node->v[j + i * n]);
    }
    printf("\n");
  }
  printf("\n");
}


static void print_tree_hodlr(struct TreeHODLR *hodlr) {
  int len_queue = 1;
  int max_depth_n = (int)pow(2, hodlr->height - 1); 

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **temp_pointer;
  queue[0] = hodlr->root;

  for (int i = 1; i < hodlr->height; i++) {
    printf("depth=%d\n", i-1);
    for (int j = 0; j < len_queue; j++) {
      printf("node=%d\nTOP RIGHT CORNER:\n", j);
      print_node_offdiagonal(&queue[j]->children[1].leaf->data.off_diagonal);

      printf("BOTTOM LEFT CORNER:\n");
      print_node_offdiagonal(&queue[j]->children[2].leaf->data.off_diagonal);
    
      next_level[2 * j] = queue[i]->children[0].internal;
      next_level[2 * j + 1] = queue[i]->children[3].internal;
    }
    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;

    len_queue *= 2;
  }

  printf("depth=MAX\n");
  for (int i = 0; i < len_queue; i++) {
    printf("TOP LEFT CORNER:\n");
    print_node_diagonal(&queue[i]->children[0].leaf->data.diagonal);

    printf("TOP RIGHT CORNER:\n");
    print_node_offdiagonal(&queue[i]->children[1].leaf->data.off_diagonal);
    
    printf("BOTTOM LEFT CORNER:\n");
    print_node_offdiagonal(&queue[i]->children[2].leaf->data.off_diagonal);
    
    printf("BOTTOM RIGHT CORNER:\n");
    print_node_diagonal(&queue[i]->children[3].leaf->data.diagonal);
  }
}


int main() {
  int m = 10;
  double svd_threshold = 0.1;
  int depth = 1;

  int idx;
  double *matrix = malloc(m * m * sizeof(double));
  for (int i = 0; i<m; i++) {
    for (int j = 0; j<m; j++) {
      idx = j + i * m;
      if (i == j) {
        matrix[idx] = 1;
      } else if (i == j+1 || i == j-1) {
        matrix[idx] = 0.5;
      } else {
        matrix[idx] = 0;
      }
    }
  }

  print_matrix(m, m, matrix);

  printf("%d x %d matrix initialised - constructing HOLDR matrix...\n", m, m);

  struct TreeHODLR *hodlr = allocate_tree(depth);
  printf("HODLR matrix allocated, converting from dense...\n");

  dense_to_tree_hodlr(hodlr, m, matrix, svd_threshold);

  printf("HODLR matrix computed, printing...\n");

  //printf("diagonal=%d, off_diagonal=%d\n", DIAGONAL, OFFDIAGONAL);
  //printf("depth=%d   child.type=%d\n", hodlr.depth, hodlr.child->type);
  print_tree_hodlr(hodlr);

  free_tree_hodlr(hodlr);
  free(matrix);

  return 0;
}
