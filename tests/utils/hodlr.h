#pragma once

#include <stdbool.h>

#include "../../include/hmat_lib/hodlr.h"


void copy_block_sizes(
  const struct TreeHODLR *src,
  struct TreeHODLR *dest,
  const bool copy_s
);


void fill_leaf_node_ints(struct TreeHODLR *hodlr, const int m, int *ss);

