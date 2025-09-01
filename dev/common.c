#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/hmat_lib/hodlr.h"

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


void log_hodlr_fill(const struct TreeHODLR *const hodlr) {
  struct HODLRInternalNode **queue = hodlr->work_queue;
  long len_queue = 1, q_next_node_density = hodlr->len_work_queue;
  long q_current_node_density = q_next_node_density;
  int idx = 0;
  queue[0] = hodlr->root;

  double means[hodlr->height];
  double stds[hodlr->height];
  double numerators[hodlr->height];
  double denominators[hodlr->height];
  double dense_ms[hodlr->height];

  double *vals = malloc(hodlr->len_work_queue * sizeof(double));

  for (int level = 1; level < hodlr->height + 1; level++) {
    q_next_node_density /= 2;

    for (int parent = 0; parent < len_queue; parent++) {
      idx = parent * q_current_node_density;

      double m = (double)queue[idx]->children[1].leaf->data.off_diagonal.m;
      double s = (double)queue[idx]->children[1].leaf->data.off_diagonal.s;
      double n = (double)queue[idx]->children[1].leaf->data.off_diagonal.n;

      dense_ms[level-1] += m * m + n * n;

      const double numerator = m * s + n * s, denominator = m * n;
      const double ratio = numerator / denominator;

      numerators[level-1] = numerator; denominators[level-1] = denominator;
      vals[parent] = ratio;
      means[level-1] += ratio;

      #ifdef _TEST_HODLR
      cr_log_info("ratio=%f (level=%d, i=%d)", ratio, level, parent);
      #else
      printf("ratio=%f (level=%d, i=%d)\n", ratio, level, parent);
      #endif

      queue[(2 * parent + 1) * q_next_node_density] = 
        queue[idx]->children[3].internal;
      queue[idx] = queue[idx]->children[0].internal;
    }

    means[level-1] /= len_queue;
    for (int i = 0; i < len_queue; i++) {
      const double diff = fabs(vals[i] - means[level-1]);
      stds[level-1] += diff * diff;
    }
    stds[level-1] = sqrtf(stds[level-1] / len_queue);

    len_queue *= 2;
    q_current_node_density = q_next_node_density;
  }

  double cumsum_num = 0.0, cumsum_denom = 0.0;
  for (int level = 1; level < hodlr->height + 1; level++) {
    cumsum_num += numerators[level-1];
    cumsum_denom += denominators[level-1];
    const int m = dense_ms[level - 1];

    #ifdef _TEST_HODLR
    cr_log_info("level=%d -> ratio = %f +- %f (cumulative=%f, total=%f)", level, 
                means[level-1], stds[level-1], cumsum_num / cumsum_denom,
                (cumsum_num + m) / (cumsum_denom + m));
    #else
    printf("level=%d -> ratio = %f +- %f (cumulative=%f, total=%f)\n", 
           level, means[level-1], stds[level-1], cumsum_num / cumsum_denom,
           (cumsum_num + m) / (cumsum_denom + m));
    #endif
  }

  free(vals);
}

