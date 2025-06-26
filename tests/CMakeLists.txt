if(BUILD_TESTING)
  add_executable(unit_test src/test_allocators.c src/test_tree_construct.c src/test_vector_algebra.c src/test_hodlr_dense_algebra.c ../src/lapack_wrapper.c src/utils.c src/common_data.c)
    
  target_link_libraries(unit_test ${MATH_LIBRARY})
  target_link_libraries(unit_test openblas)
  target_link_libraries(unit_test criterion)

  add_test(NAME unit_test COMMAND unit_test -j1 --verbose=0)

  set_property(TEST unit_test PROPERTY
      ENVIRONMENT "CTEST_OUTPUT_ON_FAILURE=1"
  )

  if(BUILD_OPENMP_TESTS AND OpenMP_C_FOUND)
    add_executable(test_constructors_omp src/test_allocators.c src/test_tree_construct.c ../src/lapack_wrapper.c src/utils.c src/common_data.c)

    target_link_libraries(test_constructors_omp ${MATH_LIBRARY})
    target_link_libraries(test_constructors_omp openblas)
    target_link_libraries(test_constructors_omp criterion)
    target_link_libraries(test_constructors_omp OpenMP::OpenMP_C)

    add_test(NAME test_constructors_omp COMMAND test_constructors_omp -j1 --verbose=0)

    set_property(TEST test_constructors_omp PROPERTY
        ENVIRONMENT "CTEST_OUTPUT_ON_FAILURE=1"
    )
  endif()

  add_custom_target(vtest COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
endif()
