#include <stdlib.h>
#include <stdio.h>

#include "../include/internal/lapack_wrapper.h"
#include "../include/hmat_lib/error.h"

#ifdef GPU_BUILD
#include <magma_types.h>
#include <magma_d.h>

#define restrict __restrict
#endif


int svd_double(
  const int m,
  const int n,
  const int n_singular_values,
  const int matrix_ld,
  double *restrict const matrix,
  double *restrict const s,
  double *restrict const u,
  double *restrict const vt,
  int *restrict const ierr
) {
  double work_size;
  double *work = &work_size;
  int lwork = -1;
#ifdef CUDA
  int *iwork;
  cudaMalloc((void**)&iwork, 8 * n_singular_values * sizeof(double));
#else
  int *iwork = malloc(8 * n_singular_values * sizeof(int));
#endif
  if (iwork == NULL) {
    #pragma omp atomic write
    *ierr = SVD_ALLOCATION_FAILURE;
    return 0;
  }

  int info = 0;
#ifndef GPU_BUILD
  dgesdd_("S", &m, &n, matrix, &matrix_ld, s, u, &m, vt, 
          &n_singular_values, work, &lwork, iwork, &info);
#else
  magma_dgesdd(
    MagmaSomeVec, m, n, matrix, matrix_ld, s, u, m, vt, n_singular_values, 
    work, lwork, iwork, &info
  );
#endif

  if (info < 0) {
#ifdef CUDA
    cudaFree(iwork);
#else
    free(iwork);
#endif

    #pragma omp atomic write
    *ierr = SVD_FAILURE;
    return info;
  }

  lwork = (int)work_size;
#ifdef CUDA
  cudaMalloc((void**)&work, lwork * sizeof(double));
  if (work == NULL) {
    cudaFree(iwork);
#else
  work = malloc(lwork * sizeof(double));
  if (work == NULL) {
    free(iwork);
#endif

    #pragma omp atomic write
    *ierr = SVD_ALLOCATION_FAILURE;
    return info;
  }

#ifndef GPU_BUILD
  dgesdd_("S", &m, &n, matrix, &matrix_ld, s, u, &m, vt, 
          &n_singular_values, work, &lwork, iwork, &info);
#else
  magma_dgesdd(
    MagmaSomeVec, m, n, matrix, matrix_ld, s, u, m, vt, n_singular_values, 
    work, lwork, iwork, &info
  );
#endif

  if (info < 0) {
    #pragma omp atomic write
    *ierr = SVD_FAILURE;
  }

#ifdef CUDA
  cudaFree(work); cudaFree(iwork);
#else
  free(work); free(iwork);
#endif

  return info;
}
