#pragma once

int dense_to_tree_hodlr(
  struct TreeHODLR *hodlr,
  const int m,
  const int *ms,
  double *matrix,
  const double svd_threshold,
  int *ierr
#ifdef _TEST_HODLR
  , void *(*permanent_allocator)(size_t size),
  void(*free)(void *ptr)
#endif
);


