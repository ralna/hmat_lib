#include <stdio.h>
#include <stdlib.h>

int main() {
  int m = 4, n = 3;
  int n_singular_values = m < n ? m : n;

  double *matrix = malloc(m * n * sizeof(double));
  matrix[0] = 5;
  matrix[1] = 0;
  matrix[2] = 1;
  matrix[3] = 0; matrix[4] = -5; matrix[5] = 0;
  matrix[6] = 1; matrix[7] = 0; matrix[8] = 1;
  matrix[9] = 1; matrix[10] = 1; matrix[11] = 1;

  double *s = malloc(n_singular_values * sizeof(double));
  double *u = malloc(m * n_singular_values * sizeof(double));
  double *vt = malloc(n_singular_values * n * sizeof(double));

  double work_size;
  double *work = &work_size;
  int lwork = -1;
  int *iwork = malloc(8 * n_singular_values * sizeof(int));
  int info = 0;

  dgesdd_("S", &m, &n, matrix, &m, s, u, &m, vt, &n_singular_values, work, &lwork, iwork, &info);
  if (info != 0) {
    printf("something went wrong");
  
   free(iwork);
    free(s); free(u); free(vt);

    return 1;
  }

  lwork = work_size;
  work = malloc(lwork * sizeof(double));

  dgesdd_("S", &m, &n, matrix, &m, s, u, &m, vt, &n_singular_values, work, &lwork, iwork, &info);
  if (info != 0) {
    printf("something went wrong (really)");
  
    free(work); free(iwork);
    free(s); free(u); free(vt);

    return 1;
  }

  for (int i=0; i<n; i++) {
    for (int j=0; j<n_singular_values; j++) {
      printf("%f    ", u[i + j * m]);
    }
    printf("\n");
  }
  printf("\n");

  for (int i=0; i<n_singular_values; i++) {
    printf("%f    ", s[i]);
  }
  printf("\n");

  for (int i=0; i<n; i++) {
    for (int j=0; j<n_singular_values; j++) {
      printf("%f    ", vt[i + j * m]);
    }
    printf("\n");
  }
  printf("\n");


  free(work); free(iwork);
  free(s); free(u); free(vt);

  return 0;
}
