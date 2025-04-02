#include <stdlib.h>
#include <stdio.h>

#include "../include/lapack_wrapper.h"
#include "../include/error.h"


int svd_double(int m,
               int n,
               int n_singular_values,
               int matrix_leading_dim,
               double *restrict matrix,
               double *restrict s,
               double *restrict u,
               double *restrict vt,
               int *restrict ierr) {
  double work_size;
  double *work = &work_size;
  int lwork = -1;
  int *iwork = malloc(8 * n_singular_values * sizeof(int));
  if (iwork == NULL) {
    *ierr = SVD_ALLOCATION_FAILURE;
    return 0;
  }

  int info = 0;

  //printf("before dgesdd=%d\n", info);
  dgesdd_("S", &m, &n, matrix, &matrix_leading_dim, s, u, &m, vt, &n_singular_values, work, &lwork, iwork, &info);
  if (info != 0) {
    free(iwork);
    *ierr = SVD_FAILURE;
    return info;
  }
  //printf("first dgesdd completed=%d\n", info);

  lwork = work_size;
  work = malloc(lwork * sizeof(double));
  if (work == NULL) {
    free(iwork);
    *ierr = SVD_ALLOCATION_FAILURE;
    return info;
  }
  //printf("lowrk=%d\n", lwork);
  dgesdd_("S", &m, &n, matrix, &matrix_leading_dim, s, u, &m, vt, &n_singular_values, work, &lwork, iwork, &info);
  if (info != 0) {
    *ierr = SVD_FAILURE;
  }
  //printf("dgesdd completed=%d\n", info);
  free(work); free(iwork);

  return info;
}
