#include <stdlib.h>

double * read_dense_matrix(char *path, 
                           int *size,
                           void *(*malloc)(size_t size),
                           void(*free)(void *ptr));

