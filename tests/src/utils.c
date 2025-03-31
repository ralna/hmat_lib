#include <stdio.h>
#include <math.h>

#include <criterion/logging.h>
#include <criterion/criterion.h>
#include <criterion/new/assert.h>

#include "../include/utils.h"


static double DELTA = 1e-10;


void log_matrix(double *matrix, int m, int n, int lda) {
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      printf("%f    ", matrix[j * lda + i]);
    }
    printf("\n");
  }
  printf("\n");
}


struct ParametersTestCompress {
  int m_full;
  int m;
  int n;
  double svd_threshold;
  int expected_n_singular;
  double *matrix;
  double *u_expected;
  double *v_expected;
  double *full_matrix;
};


void expect_arr_double_eq(double *actual, 
                          double *expected, 
                          int m, 
                          int n,
                          int ld_actual,
                          int ld_expected) {
  int errors = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (fabs(actual[j + i * ld_actual] - expected[j + i * ld_expected]) > DELTA) {
        cr_log_error("actual value '%f' at index [%d, %d] is different from the expected '%f'", 
                     actual[j + i * ld_actual], j, i, expected[j + i * ld_expected]);
        errors += 1;
      }
    }
  }

  if (errors > 0) {
    cr_fail("The matrices are not equal (%d errors)", errors);
    cr_log_info("Actual:");
    log_matrix(actual, m, n, ld_actual);
    cr_log_info("Expected:");
    log_matrix(expected, m, n, ld_expected);
  }
}



