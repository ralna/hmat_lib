#ifndef HODLR_ERROR_H
#define HODLR_ERROR_H

/**
 * Error code enum.
 *
 * Used to communicate the reason for an error.
 */
enum ErrorCode {
  SUCCESS,
  ALLOCATION_FAILURE,
  SVD_FAILURE,
  SVD_ALLOCATION_FAILURE,
  INPUT_ERROR
};

#endif

