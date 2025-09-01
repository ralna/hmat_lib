void dgemm_(const char*, const char*, const int*,
            const int*, const int*, const double*, const double*, const int*,
            const double*, const int*, const double*, double*, const int*);


void dgemv_(const char*, const int*, const int*,
            const double*, const double*, const int*,
            const double*, const int*, const double*,
            const double*, const int*);

void dlacpy_(const char*, const int*, const int*, const double*, const int*,
             double*, const int*);

void dgeqrt_(const int*, const int*, const int*, double*, const int*, double*,
             const int*, double*, int*);

void dtrmm_(const char*, const char*, const char*, const char*, const int*,
            const int*, const double*, const double*, const int*, double*,
            const int*);

void dgemqrt_(const char*, const char*, const int*, const int*, const int*,
              const int*, const double*, const int*, const double*, 
              const int*, double*, const int*, double*, int*);

