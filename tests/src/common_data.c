#include <criterion/criterion.h>

#include "../include/common_data.h"


double * construct_laplacian_matrix(int m) {
  int idx;
  double *matrix = cr_malloc(m * m * sizeof(double));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      idx = j + i * m;
      if (i == j) {
        matrix[idx] = 2;
      } else if (i == j+1 || i == j-1) {
        matrix[idx] = -1;
      } else {
        matrix[idx] = 0;
      }
    }
  }

  return matrix;
}


double * construct_identity_matrix(int m) {
  int idx;
  double *matrix = cr_malloc(m * m * sizeof(double));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      idx = j + i * m;
      if (i == j) {
        matrix[idx] = 1;
      } else {
        matrix[idx] = 0;
      }
    }
  }

  return matrix;
}


double * construct_full_matrix(const int m, const double val) {
  int idx = 0, i = 0, j = 0;
  double *matrix = cr_malloc(m * m * sizeof(double));
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j ++) {
      matrix[i + j * m] = val;
    }
  }
  return matrix;
}

