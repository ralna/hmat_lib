#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include "../include/hmat_lib/hodlr.h"
#include "../include/hmat_lib/allocators.h"
#include "../include/hmat_lib/constructors.h"

#include "../tests/utils/common_data.h"


int main(int argc, char *argv[]) {
  const int omp_n_threads = argc > 1 ? atoi(argv[1]) : 1;

  const int n_repeats = 5;
  const double svd_threshold = 1e-8;

  enum {n_ms = 8, n_heights = 5};
  const int ms[n_ms] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

  int ierr;

  double *matrix = malloc(ms[n_ms - 1] * ms[n_ms - 1] * sizeof(double));

  for (int height = 1; height < n_heights + 1; height++) {
    for (int midx = 0; midx < n_ms; midx++) {
      const int m = ms[midx];
      for (int repeat = 0; repeat < n_repeats; repeat++) {
        double start, end;
        start = omp_get_wtime();
        struct TreeHODLR *hodlr = allocate_tree_monolithic(height, &ierr);
        end = omp_get_wtime();
        const double time_alloc = end - start;

        fill_decay_matrix_random_sorted(m, 1.0, matrix);

        start = omp_get_wtime();
        dense_to_tree_hodlr(hodlr, m, NULL, matrix, svd_threshold, &ierr);
        end = omp_get_wtime();
        const double time_svd = end - start;

        start = omp_get_wtime();
        free_tree_hodlr(&hodlr);
        end = omp_get_wtime();

        printf("omp=%d, h=%d, m=%d, r=%d, alloc=%e, svd=%e, free=%e\n", 
               omp_n_threads, height, m, repeat, time_alloc, time_svd, 
               end - start);
      }
    }
  }

  free(matrix);
}
