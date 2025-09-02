if(BUILD_TESTING)
  add_library(
    testing_utils
    utils.c common_data.c hodlr.c
  )

  target_link_libraries(testing_utils ${MATH_LIBRARY})
  target_link_libraries(testing_utils openblas)
  target_link_libraries(testing_utils criterion)
endif()
