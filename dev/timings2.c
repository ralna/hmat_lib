#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include "../include/hmat_lib/hodlr.h"
#include "../include/hmat_lib/allocators.h"
#include "../include/hmat_lib/constructors.h"
#include "../include/hmat_lib/vector_algebra.h"
#include "../include/hmat_lib/dense_algebra.h"
#include "../include/hmat_lib/hodlr_algebra.h"
#include "../include/hmat_lib/error.h"

#include "../include/internal/lapack_wrapper.h"
#include "../include/internal/blas_wrapper.h"
#include "../tests/utils/common_data.h"


static inline int compress_off_diagonal(
  struct NodeOffDiagonal *restrict const node,
  const int m_smaller,
  const int matrix_ld,
  double *restrict const lapack_matrix,
  double *restrict const s,
  double *restrict const u,
  double *restrict const vt,
  const double svd_threshold,
  int *restrict const ierr
#ifdef _TEST_HODLR
  , void *(*malloc)(size_t size)
#endif
) {
  const int m = node->m, n = node->n;
  int result = 
    svd_double(m, n, m_smaller, matrix_ld, lapack_matrix, s, u, vt, ierr);
  if (*ierr != SUCCESS) {
    return result;
  }

  int svd_cutoff_idx = 1;
  if (s[0] > svd_threshold) {
    for (svd_cutoff_idx=1; svd_cutoff_idx < m_smaller; svd_cutoff_idx++) {
      if (s[svd_cutoff_idx] < svd_threshold * s[0]) {
        break;
      }
    }
  }

  double *u_top_right = malloc(m * svd_cutoff_idx * sizeof(double));
  if (u_top_right == NULL) {
    #pragma omp atomic write
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<m; j++) {
      u_top_right[j + i * m] = u[j + i * m] * s[i];
    }
  }

  double *v_store = malloc(svd_cutoff_idx * n * sizeof(double));
  if (v_store == NULL) {
    #pragma omp atomic write
    *ierr = ALLOCATION_FAILURE;
    return result;
  }
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<n; j++) {
      v_store[j + i * n] = vt[i + j * m_smaller];
    }
  }

  node->u = u_top_right;
  node->v = v_store;
  node->s = svd_cutoff_idx;

  return result;
}


static inline struct HODLRInternalNode * create_internal(
  struct HODLRInternalNode *parent,
  const int m,
  struct HODLRLeafNode *diagonal,
  double *restrict s,
  double *restrict u,
  double *restrict vt,
  const double svd_threshold,
  int *ierr
) {
  struct HODLRInternalNode *node = malloc(sizeof(struct HODLRInternalNode));
  node->m = m;
  node->parent = parent;

  const int m_smaller = m / 2;
  const int m_larger = m - m_smaller;
  double *matrix = diagonal->data.diagonal.data;

  node->children[0].leaf = diagonal;
  node->children[0].leaf->parent = node;
  node->children[0].leaf->type = DIAGONAL;
  node->children[0].leaf->data.diagonal.m = m_larger;
  node->children[0].leaf->data.diagonal.data = 
    malloc(m_larger * m_larger * sizeof(double));
  dlacpy_("A", &m_larger, &m_larger, matrix, &m, 
          node->children[0].leaf->data.diagonal.data, &m_larger);

  node->children[1].leaf = malloc(sizeof(struct HODLRLeafNode));
  node->children[1].leaf->parent = node;
  node->children[1].leaf->type = OFFDIAGONAL;
  node->children[1].leaf->data.off_diagonal.m = m_larger;
  node->children[1].leaf->data.off_diagonal.n = m_smaller;

  compress_off_diagonal(
    &node->children[1].leaf->data.off_diagonal, m_smaller, m, 
    matrix + m_larger * m, s, u, vt, svd_threshold, ierr
  );

  node->children[2].leaf = malloc(sizeof(struct HODLRLeafNode));
  node->children[2].leaf->parent = node;
  node->children[2].leaf->type = OFFDIAGONAL;
  node->children[2].leaf->data.off_diagonal.m = m_smaller;
  node->children[2].leaf->data.off_diagonal.n = m_larger;

  compress_off_diagonal(
    &node->children[2].leaf->data.off_diagonal, m_smaller, m, 
    matrix + m_larger, s, u, vt, svd_threshold, ierr
  );

  node->children[3].leaf = malloc(sizeof(struct HODLRLeafNode));
  node->children[3].leaf->parent = node;
  node->children[3].leaf->type = DIAGONAL;
  node->children[3].leaf->data.diagonal.m = m_smaller;
  node->children[3].leaf->data.diagonal.data = 
    malloc(m_smaller * m_smaller * sizeof(double));
  dlacpy_("A", &m_smaller, &m_smaller, matrix + m_larger + m_larger * m, &m, 
          node->children[3].leaf->data.diagonal.data, &m_smaller);

  free(matrix);

  return node;
}


void increase_height(
  struct TreeHODLR *hodlr,
  const int new_height,
  const double svd_threshold,
  int *restrict const ierr
) {
  if (new_height <= hodlr->height) {
    *ierr = INPUT_ERROR;
    return;
  }

  const int n = hodlr->innermost_leaves[0]->data.diagonal.m;
  double *s = malloc((n + 2 * n * n) * sizeof(double));
  double *u = s + n;
  double *vt = u + n * n;

  long n_parent_nodes = hodlr->len_work_queue * 2;
    
  struct HODLRInternalNode **queue = realloc(
    hodlr->work_queue, n_parent_nodes * sizeof(struct HODLRInternalNode *)
  );
  struct HODLRLeafNode **leaves = realloc(
    hodlr->innermost_leaves, n_parent_nodes * 2 * sizeof(struct HODLRLeafNode *)
  );

  for (int parent = 0; parent < hodlr->len_work_queue; parent++) {
    queue[parent] = leaves[2 * parent]->parent;

    const int m_smaller = queue[parent]->m / 2;
    const int m_larger = queue[parent]->m - m_smaller;

    queue[parent]->children[0].internal = create_internal(
      queue[parent], m_larger, queue[parent]->children[0].leaf, s, u, vt,
      svd_threshold, ierr
    );
    queue[parent]->children[3].internal = create_internal(
      queue[parent], m_smaller, queue[parent]->children[3].leaf, s, u, vt,
      svd_threshold, ierr
    );
  }

  int idx = 0;
  for (int parent = 0; parent < hodlr->len_work_queue; parent++) {
    for (int child = 0; child < 4; child+=3) {
      for (int grandchild = 0; grandchild < 4; grandchild+=3) {
        leaves[idx] = 
          queue[parent]->children[child].internal->children[grandchild].leaf;
        idx++;
      }
    }
  }
  
  hodlr->height = new_height;
  hodlr->len_work_queue = n_parent_nodes;
  hodlr->work_queue = queue;
  hodlr->innermost_leaves = leaves;

  free(s);
}


static int compare_double(const void* p1, const void* p2) {
  if (*(double*)p1 < *(double*)p2) return -1;
  if (*(double*)p1 > *(double*)p2) return 1;
  return 0;
}


int main(int argc, char *argv[]) {
  const int omp_n_threads = argc > 1 ? atoi(argv[1]) : 1;

  const int n_repeats = 25;
  const double svd_threshold = 1e-8;

  enum {n_ms = 8, n_heights = 5};
  const int ms[n_ms] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
  const int max_m = ms[n_ms - 1];

  int ierr; const int inc = 1;
  const double alpha = 1.0, beta = 0.0;

  srand(42);
  double *vec = malloc(max_m * sizeof(double));
  for (int i = 0; i < max_m; i++)
    vec[i] = 1.0 - 2.0 * ((double)rand() / RAND_MAX);
  qsort(vec, max_m, sizeof(double), compare_double);

  double *matrix = malloc(max_m * max_m * sizeof(double));

  for (int midx = 0; midx < n_ms; midx++) {
    const int m = ms[midx];
    fill_decay_matrix(m, vec, 1.0, matrix);
    
    struct TreeHODLR *hodlr = allocate_tree(1, &ierr);
    dense_to_tree_hodlr(hodlr, m, NULL, matrix, svd_threshold, &ierr);
    fill_decay_matrix(m, vec, 1.0, matrix);
    
    for (int height = 1; height < n_heights + 1; height++) {
      if (height != 1) {
        increase_height(hodlr, hodlr->height+1, svd_threshold, &ierr);
      }

      double *vector_expected = malloc(m * sizeof(double));
      double *vector_actual = malloc(m * sizeof(double));
      double *matrix_expected = malloc(m * m * sizeof(double));
      double *matrix_actual = malloc(m * m * sizeof(double));
      struct TreeHODLR *hodlr_actual = allocate_tree_monolithic(height, &ierr);
      
      for (int repeat = 0; repeat < n_repeats; repeat++) {
        double start, end;

        start = omp_get_wtime();
        dgemv_("N", &m, &m, &alpha, matrix, &m, vec, &inc, &beta, 
               vector_expected, &inc);
        end = omp_get_wtime();
        const double time_dv = end - start;

        start = omp_get_wtime();
        multiply_vector(hodlr, vec, vector_actual, &ierr);
        end = omp_get_wtime();
        const double time_hv = end - start;

        start = omp_get_wtime();
        dgemm_(
          "N", "N", &m, &m, &m, &alpha, matrix, &m, matrix, &m, &beta,
          matrix_expected, &m
        );
        end = omp_get_wtime();
        const double time_dd = end - start;

        start = omp_get_wtime();
        multiply_hodlr_dense(
          hodlr, matrix, m, m, matrix_actual, m, &ierr
        );
        end = omp_get_wtime();
        const double time_hd = end - start;

        start = omp_get_wtime();
        multiply_hodlr_hodlr(hodlr, hodlr, hodlr_actual, svd_threshold, &ierr);
        end = omp_get_wtime();
        const double time_hh = end - start;

        printf("omp=%d, h=%d, m=%d, r=%d, dv=%e, hv=%e, dd=%e, hd=%e, hh=%e\n", 
               omp_n_threads, height, m, repeat, time_dv, time_hv, time_dd, 
               time_hd, time_hh);
      }
      
      free(vector_expected); free(vector_actual);
      free(matrix_expected); free(matrix_actual);
      free_tree_hodlr(&hodlr_actual);
    }

    free_tree_hodlr(&hodlr);
  }

  free(matrix); free(vec);
}
