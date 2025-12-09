if(BUILD_TESTING)
  add_library(
    testing_utils
    expect.c common_data.c hodlr.c
  )

  target_link_libraries(testing_utils ${MATH_LIBRARY})
  target_link_libraries(testing_utils BLAS::BLAS)
  target_link_libraries(testing_utils PkgConfig::CRITERION)
endif()
