#include <stdio.h>
#include <stdlib.h>

#include "tree.h"


void print_node_diagonal(struct NodeDiagonal *node) {
  int m = node->m;

  for (int i=0; i<m; i++) {
    for (int j=0; j < m; j++) {
      printf("%f    ", node->data[j * m + i]);
    }
    printf("\n");
  }
  printf("\n");
}


void print_node_offdiagonal(struct NodeOffDiagonal *node) {
  int m = node->m; int s = node->s; int n = node->n;

  printf("U:\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < s; j++) {
      printf("%f    ", node->u[j * m + i]);
    }
    printf("\n");
  }

  printf("\nV_T:\n");
  for (int i = 0; i < s; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f    ", node->v[j + i * n]);
    }
    printf("\n");
  }
  printf("\n");
}


void print_tree_hodlr(struct TreeHODLR *hodlr) {
  struct HODLRNode *node = hodlr->child;
 
  printf("%d\n", hodlr->child->type);
  print_node_diagonal(&(node->data.diagonal));

  //node = node->next_sibling;
  printf("%d\n", node->type);
  print_node_offdiagonal(&(node->data.off_diagonal));

  //node = node->next_sibling;
  printf("%d\n", node->type);
  print_node_offdiagonal(&(node->data.off_diagonal));

  //node = node->next_sibling;
  printf("%d\n", node->type);
  print_node_diagonal(&(node->data.diagonal));

  //for (int i = 0; i < hodlr->depth; i++) {
  
  
  //}
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

  printf("%d x %d matrix initialised - constructing HOLDR matrix...\n", m, m);

  struct TreeHODLR hodlr = dense_to_tree_hodlr(m, m, matrix, svd_threshold, depth);

  printf("HODLR matrix initialised, printing...\n");

  printf("diagonal=%d, off_diagonal=%d\n", DIAGONAL, OFFDIAGONAL);
  printf("depth=%d   child.type=%d\n", hodlr.depth, hodlr.child->type);
  print_tree_hodlr(&hodlr);

  free_tree_hodlr(&hodlr);

  return 0;
}
