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
        matrix[idx] = 2.0;
      } else if (i == j+1 || i == j-1) {
        matrix[idx] = -1.0;
      } else {
        matrix[idx] = 0.0;
      }
    }
  }
}


void fill_laplacian_converse_matrix(const int m, double *matrix) {
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      if (i == j) {
        matrix[i + j * m] = -1.0;
      } else if (i == j + 1 || i == j - 1) {
        matrix[i + j * m] = 2.0;
      } else {
        matrix[i + j * m] = 0.0;
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


double * construct_any_matrix(const int m, 
                              void(*matrix_func)(const int, double *)) {
  double *matrix = cr_malloc(m * m * sizeof(double));
  matrix_func(m, matrix);
  return matrix;
}


void fill_tridiag_symmetric1_matrix(const int m, double *matrix) {
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      if (i == j) matrix[i + j * m] = (double)(m - i);
      else if (i == j - 1 || i == j + 1) {
        matrix[i + j * m] = (double)(i * j);
      } else {
        matrix[i + j * m] = 0.0;
      }
    }
  }
}

