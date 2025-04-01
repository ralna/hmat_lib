#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/lapack_wrapper.h"
#include "../include/tree.h"
// #include <clapack.h>
//


static void print_matrix(int m, int n, double *matrix) {
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      printf("%f    ", matrix[j * m + i]);
    }
    printf("\n");
  }
  printf("\n");
}


double * decompress_off_diagonal(struct NodeOffDiagonal *node) {
  int idx, i, j, k;
  int m = node->m, n = node->n, s = node->s;
  double *matrix = malloc(node->m * node->n * sizeof(double));

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      idx = j + i * m;
      matrix[idx] = node->u[j] * node->v[i];

      for (k = 1; k < s; k++) {
        matrix[idx] += node->u[j + k * m] * node->v[i + k * n];
      }
    }
  }

  return matrix;
}


int compress_off_diagonal(struct NodeOffDiagonal *node,
                          int m, 
                          int n, 
                          int n_singular_values,
                          int matrix_leading_dim,
                          double *lapack_matrix,
                          double *s,
                          double *u,
                          double *vt,
                          double svd_threshold) {
  //printf("m=%d, n=%d, nsv=%d, lda=%d\n", m, n, n_singular_values, matrix_leading_dim);
  //print_matrix(matrix_leading_dim, matrix_leading_dim, lapack_matrix - 5);
  int result = svd_double(m, n, n_singular_values, matrix_leading_dim, lapack_matrix, s, u, vt);
  //printf("svd result %d\n", result);
  if (result != 0) {
    return result;
  }

  double primary_s_fraction = 1 / s[0];
  int svd_cutoff_idx;
  for (svd_cutoff_idx=1; svd_cutoff_idx < n_singular_values; svd_cutoff_idx++) {
    //printf("%f    ", s[svd_cutoff_idx]);
    if (s[svd_cutoff_idx] * primary_s_fraction < svd_threshold) {
      break;
    }
  }
  //printf("svd cut-off=%d, m=%d\n", svd_cutoff_idx, m);

  double *u_top_right = malloc(m * svd_cutoff_idx * sizeof(double));
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<m; j++) {
      //printf("i=%d, j=%d, idx=%d\n", i, j, j + i * m);
      u_top_right[j + i * m] = u[j + i * m] * s[i];
    }
  }
  //print_matrix(svd_cutoff_idx, m, u_top_right);

  double *v_top_right = malloc(svd_cutoff_idx * n * sizeof(double));
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<n; j++) {
      v_top_right[j + i * n] = vt[i + j * n];
    }
  }
  //print_matrix(n, svd_cutoff_idx, v_top_right);

  node->u = u_top_right;
  node->v = v_top_right;

  node->m = m;
  node->s = svd_cutoff_idx;
  node->n = n;

  return 0;
}



void dense_to_tree_hodlr(struct TreeHODLR *hodlr, 
                         int m,
                         double *matrix, 
                         double svd_threshold) { 
  int m_smaller = m / 2;
  int m_larger = m - m_smaller;
  
  int result, offset, n_singular_values=m_smaller, len_queue=1;
  double *sub_matrix_pointer;
  double *data;

  hodlr->root->m = m;

  double *s = malloc(n_singular_values * sizeof(double));
  double *u = malloc(m_larger * n_singular_values * sizeof(double));
  double *vt = malloc(n_singular_values * m_smaller * sizeof(double));

  //printf("m_top=%d, m_bottom=%d, n_left=%d, n_right=%d\n", m_top, m_bottom, n_left, n_right);
  int max_depth_n = (int)pow(2, hodlr->height-1); 
  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(struct HODLRInternalNode *));
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(struct HODLRInternalNode *));
  struct HODLRInternalNode **temp_pointer;

  //printf("%d\n", max_depth_n);
  
  queue[0] = hodlr->root;

  for (int i = 1; i < hodlr->height; i++) {
    offset = 0;
    for (int j = 0; j < len_queue; j++) {
      m_smaller = queue[j]->m / 2;
      m_larger = queue[j]->m - m_smaller;

      //printf("i=%d, j=%d\n", i, j);

      // Off-diagonal block in the top right corner
      sub_matrix_pointer = matrix + offset + m * (offset + m_larger);
      result = compress_off_diagonal(
        &(queue[j]->children[1].leaf->data.off_diagonal), 
        m_larger, m_smaller, m_smaller, m, 
        sub_matrix_pointer, 
        s, u, vt, svd_threshold
      );
      if (result != 0) {
        // error out
        printf("compress 1 failed\n");
      }
      
      // Off-diagonal block in the bottom left corner
      sub_matrix_pointer = matrix + m * offset + offset + m_larger;
      result = compress_off_diagonal(
        &(queue[j]->children[2].leaf->data.off_diagonal), 
        m_smaller, m_larger, n_singular_values, m,
        sub_matrix_pointer, 
        s, u, vt, svd_threshold
      );
      if (result != 0) {
        // error out
        printf("compress 2 failed\n");
      }

      offset += queue[j]->m;

      queue[j]->children[0].internal->m = m_larger;
      queue[j]->children[3].internal->m = m_smaller;

      next_level[2 * j] = queue[j]->children[0].internal;
      next_level[2 * j + 1] = queue[j]->children[3].internal;
    }

    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;

    len_queue = len_queue * 2;
  }

  offset = 0;
  for (int i = 0; i < len_queue; i++) {
    m_smaller = queue[i]->m / 2;
    m_larger = queue[i]->m - m_smaller;

    //printf("i=%d, m_larger=%d, m_smaller=%d\n", i, m_larger, m_smaller);

    // Diagonal block in the top left corner
    data = malloc(m_larger * m_larger * sizeof(double));
    for (int j = 0; j < m_larger; j++) {
      for (int k = 0; k < m_larger; k++) {
        data[k + j * m_larger] = matrix[k + offset + (j + offset) * m];
      }
    }
    queue[i]->children[0].leaf->data.diagonal.data = data;
    queue[i]->children[0].leaf->data.diagonal.m = m_larger;

    // Off-diagonal block in the top right corner
    sub_matrix_pointer = matrix + offset + m * (offset + m_larger);
    result = compress_off_diagonal(
      &(queue[i]->children[1].leaf->data.off_diagonal), 
      m_larger, m_smaller, m_smaller, m, 
      sub_matrix_pointer, 
      s, u, vt, svd_threshold
    );
    if (result != 0) {
      // error out
      printf("compress 1 failed\n");
    }
    
    // Off-diagonal block in the bottom left corner
    sub_matrix_pointer = matrix + m * offset + offset + m_larger;
    result = compress_off_diagonal(
      &(queue[i]->children[2].leaf->data.off_diagonal), 
      m_smaller, m_larger, n_singular_values, m,
      sub_matrix_pointer, 
      s, u, vt, svd_threshold
    );
    if (result != 0) {
      // error out
      printf("compress 2 failed\n");
    }

    offset += m_larger;

    // Diagonal block in the bottom right corner
    data = malloc(m_smaller * m_smaller * sizeof(double));
    for (int j = 0; j < m_smaller; j++) {
      for (int k = 0; k < m_smaller; k++) {
        data[k + j * m_smaller] = matrix[k + offset + (j + offset) * m];
      }
    } 
    queue[i]->children[3].leaf->data.diagonal.data = data;
    queue[i]->children[3].leaf->data.diagonal.m = m_smaller;

    //printf("node=%p,   data=%p\n", &queue[i]->children[3].leaf->data.diagonal, queue[i]->children[3].leaf->data.diagonal.data);
    offset += m_smaller;
  }

  free(s); free(u); free(vt); free(queue); free(next_level);
}


struct TreeHODLR* allocate_tree(int height) {
  if (height < 1) {
    // error out
  }
  int len_queue = 1;
  int max_depth_n = (int)pow(2, height - 1);
  
  struct TreeHODLR *root = malloc(sizeof(struct TreeHODLR));
  //struct HODLRInternalNode *node = (HODLRInternalNode *)malloc(sizeof(HODLRInternalNode));

  root->height = height;
  root->innermost_leaves = malloc(max_depth_n * 2 * sizeof(struct HODLRLeafNode *));

  root->root = malloc(sizeof(struct HODLRInternalNode));
  root->root->parent = NULL;

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(void *));
  struct HODLRInternalNode **temp_pointer;
  queue[0] = root->root;

  for (int i = 1; i < height; i++) {
    for (int j = 0; j < len_queue; j++) {
      queue[j]->children[1].leaf = malloc(sizeof(struct HODLRLeafNode));
      queue[j]->children[1].leaf->type = OFFDIAGONAL;
      queue[j]->children[1].leaf->parent = queue[j];

      queue[j]->children[2].leaf = malloc(sizeof(struct HODLRLeafNode));
      queue[j]->children[2].leaf->type = OFFDIAGONAL;
      queue[j]->children[2].leaf->parent = queue[j];
 
      queue[j]->children[0].internal = malloc(sizeof(struct HODLRInternalNode));
      //queue[j]->children[0].internal.type = DIAGONAL;
      queue[j]->children[0].internal->parent = queue[j];

      queue[j]->children[3].internal = malloc(sizeof(struct HODLRInternalNode));
      //queue[j]->children[3].internal.type = DIAGONAL;
      queue[j]->children[3].internal->parent = queue[j];

      next_level[2 * j] = queue[j]->children[0].internal;
      next_level[2 * j + 1] = queue[j]->children[3].internal;
    }

    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;
    
    len_queue = len_queue * 2;
  }

  for (int i = 0; i < len_queue; i++) {
    for (int j = 0; j < 4; j ++) {
      queue[i]->children[j].leaf = malloc(sizeof(struct HODLRLeafNode));
      queue[i]->children[j].leaf->parent = queue[i];
    }
    root->innermost_leaves[i * 2] = queue[i]->children[0].leaf;
    root->innermost_leaves[i * 2 + 1] = queue[i]->children[3].leaf;
    
    queue[i]->children[0].leaf->type = DIAGONAL;
    queue[i]->children[1].leaf->type = OFFDIAGONAL;
    queue[i]->children[2].leaf->type = OFFDIAGONAL;
    queue[i]->children[3].leaf->type = DIAGONAL;
  }

  free(queue); free(next_level);

  return root;
}


void free_tree_hodlr(struct TreeHODLR *hodlr) {
  int i, j, k, idx;
  int n_parent_nodes = (int)pow(2, hodlr->height - 1);

  struct HODLRInternalNode **queue = malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      free(hodlr->innermost_leaves[2 * i + j]->data.diagonal.data);
      free(hodlr->innermost_leaves[2 * i + j]);
    }
  }

  for (i = hodlr->height-1; i > 0; i--) {
    n_parent_nodes /= 2;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        free(queue[idx + k]->children[1].leaf->data.off_diagonal.u);
        free(queue[idx + k]->children[1].leaf->data.off_diagonal.v);
        free(queue[idx + k]->children[1].leaf);

        free(queue[idx + k]->children[2].leaf->data.off_diagonal.u);
        free(queue[idx + k]->children[2].leaf->data.off_diagonal.v);
        free(queue[idx + k]->children[2].leaf);
      }

      free(queue[idx]);
      queue[j] = queue[idx+1]->parent;
      free(queue[idx+1]);
    }
  }

  for (i = 1; i < 3; i++) {
    free(queue[0]->children[i].leaf->data.off_diagonal.u);
    free(queue[0]->children[i].leaf->data.off_diagonal.v);
    free(queue[0]->children[i].leaf);
  }

  free(queue[0]);
  free(hodlr);
}


static inline void multiply_diagonal_vector(struct NodeDiagonal *node,
                                            double *vector,
                                            double *out) {
  int m = node->m;
  double *data = node->data;

  for (int i = 0; i < m; i++) {
    out[i] = data[i] * vector[0];
    for (int j = 1; j < m; j++) {
      out[i] += data[i + j * m] * vector[j];
    }
  }
}


static inline void multiply_off_diagonal_vector(struct NodeOffDiagonal *node,
                                                double *vector,
                                                double *out) {
  int idx, i, j, k;
  int m = node->m, n = node->n, s = node->s;
  double decompression_val;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      decompression_val = 0;
      for (k = 0; k < s; k++) {
        decompression_val += node->u[i + k * m] * node->v[j + k * n];
      }
      out[i] += decompression_val * vector[i];
    }
  }
}


double * mulitply_vector(struct TreeHODLR *hodlr,
                         double *vector,
                         double *out) {
  if (out == NULL) {
    double *out = malloc(hodlr->root->m * sizeof(double));
  }

  int offset = 0, offset2 = 0, i, j, k, idx, len_queue = 0;
  int n_parent_nodes = (int)pow(2, hodlr->height - 1);

  struct HODLRInternalNode **queue = malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      idx = 2 * i + j;
      multiply_diagonal_vector(&(hodlr->innermost_leaves[idx]->data.diagonal), 
                               vector + offset, out + offset);
      offset += hodlr->innermost_leaves[idx]->data.diagonal.m;
    }
  }

  for (i = hodlr->height-1; i > 0; i--) {
    n_parent_nodes /= 2;
    offset = 0; offset2 = 0;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        offset += queue[idx + k]->children[1].leaf->data.off_diagonal.m;
        multiply_off_diagonal_vector(&(queue[idx + k]->children[1].leaf->data.off_diagonal),
                                    vector + offset, out + offset2);
        
        multiply_off_diagonal_vector(&(queue[idx + k]->children[2].leaf->data.off_diagonal),
                                    vector + offset2, out + offset);
      
        offset += queue[idx + k]->children[1].leaf->data.off_diagonal.n;
        offset2 = offset;
      }

      queue[j] = queue[idx+1]->parent;
    }
  }

//  for (i = 1; i < 3; i++) {
//    free(queue[0]->children[i].leaf->data.off_diagonal.u);
//    free(queue[0]->children[i].leaf->data.off_diagonal.v);
//    free(queue[0]->children[i].leaf);
//  }

  return out;
}


//int main() {
  //int m = 5, n = 5, leading_dimension = 5;
  //double *matrix;

  //struct TreeHODLR hodlr = dense_to_tree_hodlr(m, n, leading_dimension, matrix);

  // hodlr.data = (double *)malloc(10 * sizeof(double));

  //if (hodlr.data == NULL) {
  ///  printf("Data is null");
  ///}

  //return 0;
//}
