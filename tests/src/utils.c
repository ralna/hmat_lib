#include <math.h>
#include <stdio.h>
#include <string.h>

#include <criterion/criterion.h>
#include <criterion/logging.h>
#include <criterion/new/assert.h>

#include "../include/utils.h"

static double DELTA = 1e-10;


void log_matrix(const double *matrix, const int m, const int n, const int lda) {
  for (int i = 0; i < m; i++) {
    char *buffer = malloc(n * 16 * sizeof(char));
    buffer[0] = '\0';

    for (int j = 0; j < n; j++) {
      char temp[16];
      snprintf(temp, sizeof(temp), "%12.5e  ", matrix[j * lda + i]);
      strcat(buffer, temp);
      //printf("%f    ", matrix[j * lda + i]);
    }
    //printf("\n");
    cr_log_info("%s", buffer);
    free(buffer);
  }
  //printf("\n");
}


int expect_matrix_double_eq_safe(
  const double *restrict actual, 
  const double *restrict expected, 
  const int m_actual, 
  const int n_actual,
  const int m_expected, 
  const int n_expected,
  const int ld_actual, 
  const int ld_expected,
  const char name)
{
  int err = 0;
  if (m_actual != m_expected) {
    err = 1;
    cr_fail("actual matrix dimension 1 (M) different than expected (actual=%d vs expected=%d)",
            m_actual, m_expected);
  }
  
  if (n_actual != n_expected) {
    err = 1;
    cr_fail("actual matrix dimension 2 (N) different than expected (actual=%d vs expected=%d)",
            n_actual, n_expected);
  }
  if (err == 0) {
    expect_matrix_double_eq(actual, expected, m_expected, n_expected, ld_actual, ld_expected, name);
  }

  return err;
}


void expect_matrix_double_eq(const double *restrict actual, const double *restrict expected, 
                          const int m, const int n,
                          const int ld_actual, const int ld_expected,
                          const char name) {

  int errors = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (fabs(actual[j + i * ld_actual] - expected[j + i * ld_expected]) >
          DELTA) {
        cr_log_error("actual value '%f' at index [%d, %d] is different from "
                     "the expected '%f'",
                     actual[j + i * ld_actual], j, i,
                     expected[j + i * ld_expected]);
        errors += 1;
      }
    }
  }

  if (errors > 0) {
    cr_fail("The %c matrices are not equal (%d errors)", name, errors);
    cr_log_info("Actual:");
    log_matrix(actual, m, n, ld_actual);
    cr_log_info("Expected:");
    log_matrix(expected, m, n, ld_expected);
  }
}

