if(BUILD_TESTING)
  message(STATUS "Building unit tests")

  add_executable(
    unit_test 
    test_allocators.c 
    test_tree_construct.c 
    test_vector_algebra.c 
    test_hodlr_dense_algebra.c 
    test_hodlr_hodlr_algebra.c
    ../../src/utils.c
    ../../src/lapack_wrapper.c 
  )
    
  target_link_libraries(unit_test ${MATH_LIBRARY})
  target_link_libraries(unit_test openblas)
  target_link_libraries(unit_test criterion)

  target_link_libraries(unit_test testing_utils)

  add_test(NAME unit_test COMMAND unit_test -j1 --verbose=0)

  set_property(TEST unit_test PROPERTY
      ENVIRONMENT "CTEST_OUTPUT_ON_FAILURE=1"
  )

  if(BUILD_OPENMP_TESTS AND OpenMP_C_FOUND)
    message(STATUS "Building OpenMP tests")

    add_executable(
      test_constructors_omp 
      test_allocators.c 
      test_tree_construct.c 
      ../../src/utils.c
      ../../src/lapack_wrapper.c 
    )

    target_link_libraries(test_constructors_omp ${MATH_LIBRARY})
    target_link_libraries(test_constructors_omp openblas)
    target_link_libraries(test_constructors_omp criterion)
    target_link_libraries(test_constructors_omp OpenMP::OpenMP_C)

    target_link_libraries(test_constructors_omp testing_utils)

    add_test(NAME test_constructors_omp COMMAND test_constructors_omp -j1 --verbose=0)

    set_property(TEST test_constructors_omp PROPERTY
        ENVIRONMENT "CTEST_OUTPUT_ON_FAILURE=1"
    )
  endif()

  add_custom_target(vtest COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
endif()
