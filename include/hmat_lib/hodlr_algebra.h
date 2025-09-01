#pragma once


int multiply_hodlr_hodlr(
  const struct TreeHODLR *hodlr1,
  const struct TreeHODLR *hodlr2,
  struct TreeHODLR *out,
  const double svd_threshold,
  int *ierr
);

