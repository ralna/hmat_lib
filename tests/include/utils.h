
void log_matrix(const double *matrix, const int m, const int n, const int lda);

void expect_matrix_double_eq(const double *actual, 
                             const double *expected, 
                             const int m, 
                             const int n,
                             const int ld_actual,
                             const int ld_expected,
                             const char name);

int expect_matrix_double_eq_safe(
  const double *actual, 
  const double *expected, 
  const int m_actual, 
  const int n_actual,
  const int m_expected, 
  const int n_expected,
  const int ld_actual, 
  const int ld_expected,
  const char name);
 
