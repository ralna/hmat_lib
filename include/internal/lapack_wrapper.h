void dgesdd_(const char*, const int*, const int*, double*, 
             const int*, double*, double*, const int*, double*, const int*, 
             double*, const int*, int*, int*); 


int svd_double(int m,
               int n,
               int n_singular_values,
               int matrix_leading_dim,
               double *matrix,
               double *s,
               double *u,
               double *vt,
               int *ierr);

