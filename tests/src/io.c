#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>

#ifdef _TEST_HODLR
#include <criterion/logging.h>
#endif


double * read_dense_matrix(char *path, int *size) {
  FILE *fptr = fopen(path, "r");
  if (fptr == NULL) {
    #ifdef _TEST_HODLR
    char str[100];
    strerror_r(errno, &str[0], 100);
    cr_log_error("File could not be read: errno=%d (%s)", errno, str);
    #else
    char str[100];
    strerror_r(errno, &str[0], 100);
    printf("File could not be read: errno=%d (%s)\n", errno, str);
    #endif
    return NULL;
  }

  int n = 0;
  bool last_char = false;
  char try;
  while ((try = fgetc(fptr)) != '\n') {
    if (try == EOF) {
    #ifdef _TEST_HODLR
    cr_log_error("No newlines in file - matrix expected to written over "
                 "multiple lines, with each row in a separate line");
    #else
    printf("No newlines in file - matrix expected to written over multiple "
           "lines, with each row in a separate line\n");
    #endif
      fclose(fptr);
      return NULL;
    } else if (try == ' ' || try == '\t') {
      if (last_char == true) {
        n++;
        last_char = false;
      }
    } else {
      last_char = true;
    }
  }
  if (last_char == true) {
    n++;
  }

  rewind(fptr);

  double *dense = malloc(n * n * sizeof(double));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (fscanf(fptr, "%lf", &dense[i + j * n]) != 1) {
        #ifdef _TEST_HODLR
        cr_log_error("Failed to read the matrix entry at [%d, %d]", i, j);
        #else
        printf("Failed to read the matrix entry at [%d, %d]\n", i, j);
        #endif
        fclose(fptr); free(dense);
        return NULL;
      }
    }
  }

  fclose(fptr);
  *size = n;
  return dense;
}
