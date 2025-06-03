#include <criterion/criterion.h>

#include "../include/common_data.h"


double * construct_laplacian_matrix(int m) {
  double *matrix = cr_malloc(m * m * sizeof(double));

  fill_laplacian_matrix(m, matrix);

  return matrix;
}


void fill_laplacian_matrix(const int m, double *matrix) {
  int idx;
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
}


double * construct_identity_matrix(int m) {
  double *matrix = cr_malloc(m * m * sizeof(double));
  fill_identity_matrix(m, matrix);
  return matrix;
}


void fill_identity_matrix(const int m, double *matrix) {
  int idx;
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
}


double * construct_full_matrix(const int m, const double val) {
  double *matrix = cr_malloc(m * m * sizeof(double));
  fill_full_matrix(m, val, matrix);
  return matrix;
}


void fill_full_matrix(const int m, const double val, double *matrix) {
  int idx = 0, i = 0, j = 0;
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j ++) {
      matrix[i + j * m] = val;
    }
  }
}

