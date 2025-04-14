#include <stdlib.h>

#include "../include/lapack_wrapper.h"
#include "../include/flat.h"



struct FlatHODLR* dense_to_flat_hodlr(int m, double *matrix, int height, double svd_threshold) {
  struct FlatHODLR *hodlr = malloc(sizeof(struct FlatHODLR));

  int m_smaller = m / 2;
  int m_larger = m - m_smaller;

  double *s = malloc(m_smaller * sizeof(double));
  double *u = malloc(m_larger * m_smaller * sizeof(double));
  double *vt = malloc(m_smaller * m_smaller * sizeof(double));



  for (int i = 1; i < height; i++) {
    
  }

  return hodlr;
}


struct FlatHODLR de3nse_to_flat_hodlr(int m, int n, double *matrix, double svd_threshold) {
  struct FlatHODLR hodlr;

  int m_top = m / 2;
  int m_bottom = m - m_top;

  int n_left = n / 2;
  int n_right = n - n_left;

  // TOP RIGHT QUARTER
  double *lapack_matrix = malloc(m_top * n_right * sizeof(double));
  for (int j = n_left; j < n; j++) {
    for (int i = 0; i < m_top; i++) {
      lapack_matrix[i + j * m_top] = matrix[i + j * m];
    }
  }
  int n_singular_values = m_top < n_right ? m_top : n_right;
  double *s = malloc(n_singular_values * sizeof(double));
  double *u = malloc(m_top * n_singular_values * sizeof(double));
  double *vt = malloc(n_singular_values * n_right * sizeof(double));
  int result = svd_double(m_top, n_right, n_singular_values, lapack_matrix, s, u, vt);
  if (result != 0) {
    // Error out
  }

  double primary_s_fraction = 1 / s[0];
  int svd_cutoff_idx;
  for (svd_cutoff_idx=1; svd_cutoff_idx < n_singular_values; svd_cutoff_idx++) {
    if (s[svd_cutoff_idx] * primary_s_fraction < svd_threshold) {
      break;
    }
  }

  double *u_top_right = malloc(m_top * svd_cutoff_idx * sizeof(double));
  for (int i=0; i<svd_cutoff_idx; i++) {
    for (int j=0; j<m_top; i++) {
      u_top_right[j + i * m_top] = u[j + i * m_top] * s[i];
    }
  }

  double *v_top_right = malloc(svd_cutoff_idx * n_right * sizeof(double));
  for (int i=0; i<n_right; i++) {
    for (int j=0; j<svd_cutoff_idx; j++) {
      v_top_right[i + j * n_right] = vt[j + i * svd_cutoff_idx];
    }
  }

  return hodlr;
}
