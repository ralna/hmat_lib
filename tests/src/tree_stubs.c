#include <stdio.h>
#include <math.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include "../include/tree_stubs.h"

#include "../../include/error.h"
#include "../../include/lapack_wrapper.h"
#include "../../include/tree.h"


int compress_off_diagonal_cr(struct NodeOffDiagonal *restrict node,
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

  double *u_top_right = cr_malloc(m * svd_cutoff_idx * sizeof(double));
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

  double *v_top_right = cr_malloc(svd_cutoff_idx * n * sizeof(double));
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



int dense_to_tree_hodlr_cr(struct TreeHODLR *restrict hodlr, 
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

  double *s = cr_malloc(n_singular_values * sizeof(double));
  if (s == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return 0;
  }
  double *u = cr_malloc(m_larger * n_singular_values * sizeof(double));
  if (u == NULL) {
    *ierr = ALLOCATION_FAILURE;
    cr_free(s);
    return 0;
  }

  double *vt = cr_malloc(n_singular_values * m_larger * sizeof(double));
  if (vt == NULL) {
    *ierr = ALLOCATION_FAILURE;
    cr_free(s); cr_free(u);
    return 0;
  }
  int max_depth_n = (int)pow(2, hodlr->height-1); 
  struct HODLRInternalNode **queue_memory = cr_malloc(2 * max_depth_n * sizeof(struct HODLRInternalNode *));
  if (queue_memory == NULL) {
    *ierr = ALLOCATION_FAILURE;
    cr_free(s); cr_free(u); cr_free(vt);
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
      result = compress_off_diagonal_cr(
        &(queue[j]->children[1].leaf->data.off_diagonal), 
        m_larger, m_smaller, m_smaller, m, 
        sub_matrix_pointer,
        s, u, vt, svd_threshold, ierr
      );
      if (*ierr != SUCCESS) {
        //handle_error(ierr, result);  // 
        // error out
        cr_free(s); cr_free(u); cr_free(vt); cr_free(queue); cr_free(next_level);
        return result;
      }
      
      // Off-diagonal block in the bottom left corner
      sub_matrix_pointer = matrix + m * offset + offset + m_larger;
      result = compress_off_diagonal_cr(
        &(queue[j]->children[2].leaf->data.off_diagonal), 
        m_smaller, m_larger, m_smaller, m,
        sub_matrix_pointer, 
        s, u, vt, svd_threshold, ierr
      );
      if (*ierr != SUCCESS) {
        // error out
        cr_free(s); cr_free(u); cr_free(vt); cr_free(queue); cr_free(next_level);
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
    data = cr_malloc(m_larger * m_larger * sizeof(double));
    if (data == NULL) {
      *ierr = ALLOCATION_FAILURE;
      cr_free(s); cr_free(u); cr_free(vt); cr_free(queue); cr_free(next_level);
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
    result = compress_off_diagonal_cr(
      &(queue[i]->children[1].leaf->data.off_diagonal), 
      m_larger, m_smaller, m_smaller, m, 
      sub_matrix_pointer, 
      s, u, vt, svd_threshold, ierr
    );
    if (*ierr != SUCCESS) {
      // error out
      cr_free(s); cr_free(u); cr_free(vt); cr_free(queue); cr_free(next_level);
      return result;
    }
    
    // Off-diagonal block in the bottom left corner
    sub_matrix_pointer = matrix + m * offset + offset + m_larger;
    result = compress_off_diagonal_cr(
      &(queue[i]->children[2].leaf->data.off_diagonal), 
      m_smaller, m_larger, m_smaller, m,
      sub_matrix_pointer, 
      s, u, vt, svd_threshold, ierr
    );
    if (*ierr != SUCCESS) {
      // error out
      cr_free(s); cr_free(u); cr_free(vt); cr_free(queue); cr_free(next_level);
      return result;
    }

    offset += m_larger;

    // Diagonal block in the bottom right corner
    data = cr_malloc(m_smaller * m_smaller * sizeof(double));
    if (data == NULL) {
      *ierr = ALLOCATION_FAILURE;
      cr_free(s); cr_free(u); cr_free(vt); cr_free(queue); cr_free(next_level);
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
  cr_free(s); cr_free(u); cr_free(vt); cr_free(queue_memory); 

  *ierr = SUCCESS;
  
  return 0;
}



 void free_partial_tree_hodlr_cr(struct TreeHODLR *hodlr, 
                                    struct HODLRInternalNode **queue, 
                                    struct HODLRInternalNode **next_level) {
  int len_queue = 1;
  if (hodlr->root == NULL) {
    return;
  }
  queue[0] = hodlr->root;
  cr_free(hodlr->innermost_leaves);

  struct HODLRInternalNode **temp_pointer = NULL;

  for (int i = 1; i < hodlr->height; i++) {
    for (int j = 0; j < len_queue; j++) {
      if (queue[j]->children[1].leaf == NULL) {
        cr_free(queue[j]);
        return;
      }
      cr_free(queue[j]->children[1].leaf);

      if (queue[j]->children[2].leaf == NULL) {
        cr_free(queue[j]);
        return;
      }
      cr_free(queue[j]->children[2].leaf);

      if (queue[j]->children[0].internal == NULL) {
        cr_free(queue[j]);
        return;
      }
      next_level[2 * j] = queue[j]->children[0].internal;
      
      if (queue[j]->children[3].internal == NULL) {
        cr_free(queue[j]);
        return;
      }
      next_level[2 * j + 1] = queue[j]->children[3].internal;
      
      cr_free(queue[j]);
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
      cr_free(queue[i]->children[j].leaf);
    }
    cr_free(queue[i]);
  }
  cr_free(hodlr);
  hodlr = NULL;
}



struct TreeHODLR* allocate_tree_cr(const int height, int *ierr) {
  if (height < 1) {
    *ierr = INPUT_ERROR;
    return NULL;
  }
  int len_queue = 1;
  const int max_depth_n = (int)pow(2, height - 1);
  
  struct TreeHODLR *root = cr_malloc(sizeof(struct TreeHODLR));
  if (root == NULL) {
    *ierr = ALLOCATION_FAILURE;
    return NULL;
  }
  //struct HODLRInternalNode *node = (HODLRInternalNode *)cr_malloc(sizeof(HODLRInternalNode));

  root->height = height;
  root->innermost_leaves = cr_malloc(max_depth_n * 2 * sizeof(struct HODLRLeafNode *));
  if (root->innermost_leaves == NULL) {
    *ierr = ALLOCATION_FAILURE;
    cr_free(root);
    return NULL;
  }

  root->root = cr_malloc(sizeof(struct HODLRInternalNode));
  root->root->parent = NULL;

  struct HODLRInternalNode **queue = cr_malloc(max_depth_n * sizeof(void *));
  if (queue == NULL) {
    *ierr = ALLOCATION_FAILURE;
    cr_free(root->root); cr_free(root);
    return NULL;
  }
  struct HODLRInternalNode **next_level = cr_malloc(max_depth_n * sizeof(void *));
  if (next_level == NULL) {
    *ierr = ALLOCATION_FAILURE;
    cr_free(root->root); cr_free(root); cr_free(queue);
    return NULL;
  } 
  struct HODLRInternalNode **temp_pointer = NULL;
  queue[0] = root->root;

  for (int i = 1; i < height; i++) {
    for (int j = 0; j < len_queue; j++) {
      // WARNING: If the order of these cr_mallocs changes the change MUST be reflected
      // in free_partial_tree_hodlr_cr!
      queue[j]->children[1].leaf = cr_malloc(sizeof(struct HODLRLeafNode));
      if (queue[j]->children[1].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr_cr(root, queue, next_level);
        cr_free(queue); cr_free(next_level);
        return NULL;
      }
      queue[j]->children[1].leaf->type = OFFDIAGONAL;
      queue[j]->children[1].leaf->parent = queue[j];

      queue[j]->children[2].leaf = cr_malloc(sizeof(struct HODLRLeafNode));
      if (queue[j]->children[2].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr_cr(root, queue, next_level);
        cr_free(queue); cr_free(next_level);
        return NULL;
      }
      queue[j]->children[2].leaf->type = OFFDIAGONAL;
      queue[j]->children[2].leaf->parent = queue[j];

      queue[j]->children[0].internal = cr_malloc(sizeof(struct HODLRInternalNode));
      if (queue[j]->children[0].internal == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr_cr(root, queue, next_level);
        cr_free(queue); cr_free(next_level);
        return NULL;
      }
      //queue[j]->children[0].internal.type = DIAGONAL;
      queue[j]->children[0].internal->parent = queue[j];

      queue[j]->children[3].internal = cr_malloc(sizeof(struct HODLRInternalNode));
      if (queue[j]->children[3].internal == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr_cr(root, queue, next_level);
        cr_free(queue); cr_free(next_level);
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
      queue[i]->children[j].leaf = cr_malloc(sizeof(struct HODLRLeafNode));
      if (queue[i]->children[j].leaf == NULL) {
        *ierr = ALLOCATION_FAILURE;
        free_partial_tree_hodlr_cr(root, queue, next_level);
        cr_free(queue); cr_free(next_level);
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

  cr_free(queue); cr_free(next_level);

  *ierr = SUCCESS;
  return root;
}



void free_tree_data_cr(struct TreeHODLR *hodlr) {
  if (hodlr == NULL) {
    return;
  }
  int i = 0, j = 0, k = 0, idx = 0;
  int n_parent_nodes = (int)pow(2, hodlr->height - 1);

  struct HODLRInternalNode **queue = cr_malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      cr_free(hodlr->innermost_leaves[2 * i + j]->data.diagonal.data);
      hodlr->innermost_leaves[2 * i + j]->data.diagonal.data = NULL;
    }
  }

  for (i = hodlr->height-1; i > 0; i--) {
    n_parent_nodes /= 2;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        idx += k;
        cr_free(queue[idx]->children[1].leaf->data.off_diagonal.u);
        cr_free(queue[idx]->children[1].leaf->data.off_diagonal.v);
        queue[idx]->children[1].leaf->data.off_diagonal.u = NULL;
        queue[idx]->children[1].leaf->data.off_diagonal.v = NULL;
        
        cr_free(queue[idx]->children[2].leaf->data.off_diagonal.u);
        cr_free(queue[idx]->children[2].leaf->data.off_diagonal.v);
        queue[idx]->children[2].leaf->data.off_diagonal.u = NULL;
        queue[idx]->children[2].leaf->data.off_diagonal.v = NULL;
      }

      queue[j] = queue[idx+1]->parent;
    }
  }

  for (i = 1; i < 3; i++) {
    cr_free(queue[0]->children[i].leaf->data.off_diagonal.u);
    cr_free(queue[0]->children[i].leaf->data.off_diagonal.v);
    queue[0]->children[i].leaf->data.off_diagonal.u = NULL;
    queue[0]->children[i].leaf->data.off_diagonal.v = NULL;
  }
  cr_free(queue);
}



void free_tree_hodlr_cr(struct TreeHODLR **hodlr_ptr) {
  struct TreeHODLR *hodlr = *hodlr_ptr;

  int i = 0, j = 0, k = 0, idx = 0;
  if (hodlr == NULL) {
    return;
  }
  int n_parent_nodes = (int)pow(2, hodlr->height - 1);

  struct HODLRInternalNode **queue = cr_malloc(n_parent_nodes * sizeof(struct HODLRInternalNode *));

  for (i = 0; i < n_parent_nodes; i++) {
    queue[i] = hodlr->innermost_leaves[2 * i]->parent;

    for (j = 0; j < 2; j++) {
      cr_free(hodlr->innermost_leaves[2 * i + j]->data.diagonal.data);
      cr_free(hodlr->innermost_leaves[2 * i + j]);
      hodlr->innermost_leaves[2 * i + j] = NULL;
    }
  }
  cr_free(hodlr->innermost_leaves);
  hodlr->innermost_leaves = NULL;

  for (i = hodlr->height-1; i > 0; i--) {
    n_parent_nodes /= 2;

    for (j = 0; j < n_parent_nodes; j++) {
      idx = 2 * j;
      for (k = 0; k < 2; k++) {
        cr_free(queue[idx + k]->children[1].leaf->data.off_diagonal.u);
        cr_free(queue[idx + k]->children[1].leaf->data.off_diagonal.v);
        cr_free(queue[idx + k]->children[1].leaf);
        queue[idx + k]->children[1].leaf = NULL;

        cr_free(queue[idx + k]->children[2].leaf->data.off_diagonal.u);
        cr_free(queue[idx + k]->children[2].leaf->data.off_diagonal.v);
        cr_free(queue[idx + k]->children[2].leaf);
        queue[idx + k]->children[1].leaf = NULL;
      }

      cr_free(queue[idx]);
      queue[idx] = NULL;
      queue[j] = queue[idx+1]->parent;
      cr_free(queue[idx+1]);
      queue[idx+1] = NULL;
    }
  }

  for (i = 1; i < 3; i++) {
    cr_free(queue[0]->children[i].leaf->data.off_diagonal.u);
    cr_free(queue[0]->children[i].leaf->data.off_diagonal.v);
    cr_free(queue[0]->children[i].leaf);
    queue[0]->children[i].leaf = NULL;
  }

  cr_free(queue[0]);
  cr_free(hodlr);
  *hodlr_ptr = NULL;
  cr_free(queue);
}


