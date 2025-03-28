#include <stdlib.h>
#include <stdio.h>

#include "../include/lapack_wrapper.h"


int svd_double(int m,
               int n,
               int n_singular_values,
               int matrix_leading_dim,
               double *matrix,
               double *s,
               double *u,
               double *vt) {
  double work_size;
  double *work = &work_size;
  int lwork = -1;
  int *iwork = malloc(8 * n_singular_values * sizeof(int));
  if (iwork == NULL) {
    return 1;
  }

  int info = 0;

  //printf("before dgesdd=%d\n", info);
  dgesdd_("S", &m, &n, matrix, &matrix_leading_dim, s, u, &m, vt, &n_singular_values, work, &lwork, iwork, &info);
  if (info != 0) {
    free(iwork);

    return info;
  }
  //printf("first dgesdd completed=%d\n", info);

  lwork = work_size;
  work = malloc(lwork * sizeof(double));
  if (work == NULL) {
    free(iwork);
    return 1;
  }
  //printf("lowrk=%d\n", lwork);
  dgesdd_("S", &m, &n, matrix, &matrix_leading_dim, s, u, &m, vt, &n_singular_values, work, &lwork, iwork, &info);

  //printf("dgesdd completed=%d\n", info);
  free(work); free(iwork);

  return info;
}
