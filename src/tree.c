#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/lapack_wrapper.h"
#include "../include/blas_wrapper.h"
#include "../include/tree.h"
#include "../include/error.h"
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


int compress_off_diagonal(struct NodeOffDiagonal *restrict node,
                          int m, 
                          int n, 
                          int n_singular_values,
                          int matrix_leading_dim,
                          double *restrict lapack_matrix,
                          double *restrict s,
                          double *restrict u,
                          double *restrict vt,
                          double svd_threshold,
                          int *restrict ierr) {
  //printf("m=%d, n=%d, nsv=%d, lda=%d\n", m, n, n_singular_values, matrix_leading_dim);
  //print_matrix(matrix_leading_dim, matrix_leading_dim, lapack_matrix - 5);
  int result = svd_double(m, n, n_singular_values, matrix_leading_dim, lapack_matrix, s, u, vt, ierr);
  //printf("svd result %d\n", result);
  if (*ierr != SUCCESS) {
    return result;
  }

  int svd_cutoff_idx = 1;
  for (svd_cutoff_idx=1; svd_cutoff_idx < n_singular_values; svd_cutoff_idx++) {
    //printf("%f    ", s[svd_cutoff_idx]);
    if (s[svd_cutoff_idx] < svd_threshold * s[0]) {
      break;
    }
  }
  //printf("svd cut-off=%d, m=%d\n", svd_cutoff_idx, m);

  double *u_top_right = malloc(m * svd_cutoff_idx * sizeof(double));
  if (u_top_right == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<m; j++) {
      //printf("i=%d, j=%d, idx=%d\n", i, j, j + i * m);
      u_top_right[j + i * m] = u[j + i * m] * s[i];
    }
  }
  //print_matrix(svd_cutoff_idx, m, u_top_right);

  double *v_top_right = malloc(svd_cutoff_idx * n * sizeof(double));
  if (v_top_right == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<n; j++) {
      v_top_right[j + i * n] = vt[i + j * n_singular_values];
    }
  }
  //print_matrix(n, svd_cutoff_idx, v_top_right);

  node->u = u_top_right;
  node->v = v_top_right;

  node->m = m;
  node->s = svd_cutoff_idx;
  node->n = n;

  *ierr = SUCCESS;
  return result;
}



int dense_to_tree_hodlr(struct TreeHODLR *restrict hodlr, 
                        int m,
                        double *restrict matrix, 
                        double svd_threshold,
                        int *ierr) {
  if (hodlr == NULL) {
    *ierr = INPUT_ERROR;
    return 0;
  }
  int m_smaller = m / 2;
  int m_larger = m - m_smaller;
  
  int result = 0, offset = 0, n_singular_values=m_smaller, len_queue=1;
  double *sub_matrix_pointer = NULL; double *data = NULL;

  hodlr->root->m = m;

  double *s = malloc(n_singular_values * sizeof(double));
  if (s == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return 0;
  }
  double *u = malloc(m_larger * n_singular_values * sizeof(double));
  if (u == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(s);
    return 0;
  }

  double *vt = malloc(n_singular_values * m_larger * sizeof(double));
  if (vt == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(s); free(u);
    return 0;
  }
  int max_depth_n = (int)pow(2, hodlr->height-1); 
  struct HODLRInternalNode **queue_memory = malloc(2 * max_depth_n * sizeof(struct HODLRInternalNode *));
  if (queue_memory == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(s); free(u); free(vt);
    return 0;
  }

  struct HODLRInternalNode **queue = queue_memory;
  struct HODLRInternalNode **next_level = queue_memory + max_depth_n;
  struct HODLRInternalNode **temp_pointer = NULL;

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
        s, u, vt, svd_threshold, ierr
      );
      if (*ierr != SUCCESS) {
        //handle_error(ierr, result);  // 
        // error out
        free(s); free(u); free(vt); free(queue); free(next_level);
        return result;
      }
      
      // Off-diagonal block in the bottom left corner
      sub_matrix_pointer = matrix + m * offset + offset + m_larger;
      result = compress_off_diagonal(
        &(queue[j]->children[2].leaf->data.off_diagonal), 
        m_smaller, m_larger, m_smaller, m,
        sub_matrix_pointer, 
        s, u, vt, svd_threshold, ierr
      );
      if (*ierr != SUCCESS) {
        // error out
        free(s); free(u); free(vt); free(queue); free(next_level);
        return result;
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
    if (data == NULL) {
      *ierr = ALLOCATION_FAILURE;
      free(s); free(u); free(vt); free(queue); free(next_level);
      return 0;
    }
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
      s, u, vt, svd_threshold, ierr
    );
    if (*ierr != SUCCESS) {
      // error out
      free(s); free(u); free(vt); free(queue); free(next_level);
      return result;
    }
    
    // Off-diagonal block in the bottom left corner
    sub_matrix_pointer = matrix + m * offset + offset + m_larger;
    result = compress_off_diagonal(
      &(queue[i]->children[2].leaf->data.off_diagonal), 
      m_smaller, m_larger, m_smaller, m,
      sub_matrix_pointer, 
      s, u, vt, svd_threshold, ierr
    );
    if (*ierr != SUCCESS) {
      // error out
      free(s); free(u); free(vt); free(queue); free(next_level);
      return result;
    }

    offset += m_larger;

    // Diagonal block in the bottom right corner
    data = malloc(m_smaller * m_smaller * sizeof(double));
    if (data == NULL) {
      *ierr = ALLOCATION_FAILURE;
      free(s); free(u); free(vt); free(queue); free(next_level);
      return 0;
    }
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
  free(s); free(u); free(vt); free(queue_memory); 

  *ierr = SUCCESS;
  
  return 0;
}


struct TreeHODLR* allocate_tree(const int height, int *ierr) {
  if (height < 1) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  int len_queue = 1;
  const int max_depth_n = (int)pow(2, height - 1);
  
  struct TreeHODLR *root = malloc(sizeof(struct TreeHODLR));
  if (root == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return NULL;
  }
  //struct HODLRInternalNode *node = (HODLRInternalNode *)malloc(sizeof(HODLRInternalNode));

  root->height = height;
  root->innermost_leaves = malloc(max_depth_n * 2 * sizeof(struct HODLRLeafNode *));
  if (root->innermost_leaves == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(root);
    return NULL;
  }

  root->root = malloc(sizeof(struct HODLRInternalNode));
  root->root->parent = NULL;

  struct HODLRInternalNode **queue = malloc(max_depth_n * sizeof(void *));
  if (queue == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(root->root); free(root);
    return NULL;
  }
  struct HODLRInternalNode **next_level = malloc(max_depth_n * sizeof(void *));
  if (next_level == NULL) {
    *ierr = ALLOCATION_FAILURE;
    free(root->root); free(root); free(queue);
    return NULL;
  } 
  struct HODLRInternalNode **temp_pointer = NULL;
  queue[0] = root->root;

  for (int i = 1; i < height; i++) {
    for (int j = 0; j < len_queue; j++) {
      // WARNING: If the order of these mallocs changes the change MUST be reflected
      // in free_partial_tree_hodlr!
      queue[j]->children[1].leaf = malloc(sizeof(struct HODLRLeafNode));
      if (queue[j]->children[1].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      queue[j]->children[1].leaf->type = OFFDIAGONAL;
      queue[j]->children[1].leaf->parent = queue[j];

      queue[j]->children[2].leaf = malloc(sizeof(struct HODLRLeafNode));
      if (queue[j]->children[2].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      queue[j]->children[2].leaf->type = OFFDIAGONAL;
      queue[j]->children[2].leaf->parent = queue[j];

      queue[j]->children[0].internal = malloc(sizeof(struct HODLRInternalNode));
      if (queue[j]->children[0].internal == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      //queue[j]->children[0].internal.type = DIAGONAL;
      queue[j]->children[0].internal->parent = queue[j];

      queue[j]->children[3].internal = malloc(sizeof(struct HODLRInternalNode));
      if (queue[j]->children[3].internal == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
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
    for (int j = 0; j < 4; j++) {
      queue[i]->children[j].leaf = malloc(sizeof(struct HODLRLeafNode));
      if (queue[i]->children[j].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr(root, queue, next_level);
        free(queue); free(next_level);
        return NULL;
      }
      
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

  *ierr = SUCCESS;
  return root;
}


static void free_partial_tree_hodlr(struct TreeHODLR *hodlr, 
                                    struct HODLRInternalNode **queue, 
                                    struct HODLRInternalNode **next_level) {
  int len_queue = 1;
  if (hodlr->root == NULL) {
    return;
  }
  queue[0] = hodlr->root;
  free(hodlr->innermost_leaves);

  struct HODLRInternalNode **temp_pointer = NULL;

  for (int i = 1; i < hodlr->height; i++) {
    for (int j = 0; j < len_queue; j++) {
      if (queue[j]->children[1].leaf == NULL) {
        free(queue[j]);
        return;
      }
      free(queue[j]->children[1].leaf);

      if (queue[j]->children[2].leaf == NULL) {
        free(queue[j]);
        return;
      }
      free(queue[j]->children[2].leaf);

      if (queue[j]->children[0].internal == NULL) {
        free(queue[j]);
        return;
      }
      next_level[2 * j] = queue[j]->children[0].internal;
      
      if (queue[j]->children[3].internal == NULL) {
        free(queue[j]);
        return;
      }
      next_level[2 * j + 1] = queue[j]->children[3].internal;
      
      free(queue[j]);
    }
    temp_pointer = queue;
    queue = next_level;
    next_level = temp_pointer;
    
    len_queue = len_queue * 2;
  }

  for (int i = 0; i < len_queue; i++) {
    for (int j = 0; j < 4; j++) {
      if (queue[i]->children[j].leaf == NULL) {
        return;
      }
      free(queue[i]->children[j].leaf);
    }
    free(queue[i]);
  }
  free(hodlr);
  hodlr = NULL;
}


void free_tree_data(struct TreeHODLR *hodlr) {
  if (hodlr == NULL) {
    return;
  }
  int i = 0, j = 0, k = 0, idx = 0;
  int n_parent_nodes = (int)pow(2, hodlr->height - 1);

  struct HODLRInternalNode **queue = malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      free(hodlr->innermost_leaves[2 * i + j]->data.diagonal.data);
      hodlr->innermost_leaves[2 * i + j]->data.diagonal.data = NULL;
    }
  }

  for (i = hodlr->height-1; i > 0; i--) {
    n_parent_nodes /= 2;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        idx += k;
        free(queue[idx]->children[1].leaf->data.off_diagonal.u);
        free(queue[idx]->children[1].leaf->data.off_diagonal.v);
        queue[idx]->children[1].leaf->data.off_diagonal.u = NULL;
        queue[idx]->children[1].leaf->data.off_diagonal.v = NULL;
        
        free(queue[idx]->children[2].leaf->data.off_diagonal.u);
        free(queue[idx]->children[2].leaf->data.off_diagonal.v);
        queue[idx]->children[2].leaf->data.off_diagonal.u = NULL;
        queue[idx]->children[2].leaf->data.off_diagonal.v = NULL;
      }

      queue[j] = queue[idx+1]->parent;
    }
  }

  for (i = 1; i < 3; i++) {
    free(queue[0]->children[i].leaf->data.off_diagonal.u);
    free(queue[0]->children[i].leaf->data.off_diagonal.v);
    queue[0]->children[i].leaf->data.off_diagonal.u = NULL;
    queue[0]->children[i].leaf->data.off_diagonal.v = NULL;
  }
  free(queue);
}


void free_tree_hodlr(struct TreeHODLR **hodlr_ptr) {
  struct TreeHODLR *hodlr = *hodlr_ptr;

  int i = 0, j = 0, k = 0, idx = 0;
  if (hodlr == NULL) {
    return;
  }
  int n_parent_nodes = (int)pow(2, hodlr->height - 1);

  struct HODLRInternalNode **queue = malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      free(hodlr->innermost_leaves[2 * i + j]->data.diagonal.data);
      free(hodlr->innermost_leaves[2 * i + j]);
      hodlr->innermost_leaves[2 * i + j] = NULL;
    }
  }
  free(hodlr->innermost_leaves);
  hodlr->innermost_leaves = NULL;

  for (i = hodlr->height-1; i > 0; i--) {
    n_parent_nodes /= 2;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        free(queue[idx + k]->children[1].leaf->data.off_diagonal.u);
        free(queue[idx + k]->children[1].leaf->data.off_diagonal.v);
        free(queue[idx + k]->children[1].leaf);
        queue[idx + k]->children[1].leaf = NULL;

        free(queue[idx + k]->children[2].leaf->data.off_diagonal.u);
        free(queue[idx + k]->children[2].leaf->data.off_diagonal.v);
        free(queue[idx + k]->children[2].leaf);
        queue[idx + k]->children[1].leaf = NULL;
      }

      free(queue[idx]);
      queue[idx] = NULL;
      queue[j] = queue[idx+1]->parent;
      free(queue[idx+1]);
      queue[idx+1] = NULL;
    }
  }

  for (i = 1; i < 3; i++) {
    free(queue[0]->children[i].leaf->data.off_diagonal.u);
    free(queue[0]->children[i].leaf->data.off_diagonal.v);
    free(queue[0]->children[i].leaf);
    queue[0]->children[i].leaf = NULL;
  }

  free(queue[0]);
  free(hodlr);
  *hodlr_ptr = NULL;
  free(queue);
}


static inline void multiply_off_diagonal_vector(
  const struct HODLRInternalNode *restrict parent,
  const double *restrict vector,
  double *restrict out,
  double *restrict workspace,
  double *restrict workspace2,
  double alpha,
  double beta,
  const int increment,
  int *restrict offset_ptr,
  const int offset2
) {
  int i = 0;
  int m = parent->children[1].leaf->data.off_diagonal.m;
  int n = parent->children[1].leaf->data.off_diagonal.n;
  int s = parent->children[1].leaf->data.off_diagonal.s;
  
  *offset_ptr += m;
  int offset = *offset_ptr;

  dgemv_("T", &n, &s, &alpha, 
          parent->children[1].leaf->data.off_diagonal.v, 
          &n, vector + offset, &increment, 
          &beta, workspace, &increment);

  dgemv_("N", &m, &s, &alpha, 
          parent->children[1].leaf->data.off_diagonal.u, 
          &m, workspace, &increment,
          &beta, workspace2, &increment);

  for (i = 0; i < m; i++) {
    out[offset2 + i] += workspace2[i];
  }
  
  s = parent->children[2].leaf->data.off_diagonal.s;
  dgemv_("T", &m, &s, &alpha, 
          parent->children[2].leaf->data.off_diagonal.v, 
          &m, vector + offset2, &increment, 
          &beta, workspace, &increment);

  dgemv_("N", &n, &s, &alpha, 
          parent->children[2].leaf->data.off_diagonal.u, 
          &n, workspace, &increment,
          &beta, workspace2, &increment);

  for (i = 0; i < n; i++) {
    out[offset + i] += workspace2[i];
  }
  *offset_ptr += n;
}


double * multiply_vector(struct TreeHODLR *hodlr,
                         double *vector,
                         double *out) {
  if (hodlr == NULL) {
    return NULL;
  }
  if (out == NULL) {
    out = malloc(hodlr->root->m * sizeof(double));
  }

  int offset = 0, offset2 = 0, i=0, j=0, k=0, idx=0;
  int m = 0, increment = 1;
  int n_parent_nodes = (int)pow(2, hodlr->height - 1);
  double alpha = 1, beta = 0;

  int largest_m = (hodlr->root->m - hodlr->root->m / 2);
  double *workspace = malloc(2 * largest_m * sizeof(double));
  double *workspace2 = workspace + largest_m;

  struct HODLRInternalNode **queue = malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      idx = 2 * i + j;
      m = hodlr->innermost_leaves[idx]->data.diagonal.m;
      dgemv_("N", &m, &m, &alpha, 
             hodlr->innermost_leaves[idx]->data.diagonal.data, 
             &m, vector + offset, &increment, 
             &beta, out + offset, &increment);
      offset += m;
    }
  }

  for (int _ = hodlr->height-1; _ > 0; _--) {
    n_parent_nodes /= 2;
    offset = 0; offset2 = 0;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        multiply_off_diagonal_vector(
          queue[idx], vector, out, workspace, workspace2, 
          alpha, beta, increment, &offset, offset2
        );
        offset2 = offset;

        idx += 1;
      }

      queue[j] = queue[2 * j + 1]->parent;
    }
  }

  offset = 0; offset2 = 0;
  multiply_off_diagonal_vector(
    hodlr->root, vector, out, workspace, workspace2, 
    alpha, beta, increment, &offset, offset2
  );
        
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
