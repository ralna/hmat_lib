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

  if(BUILD_REAL_DATA_TESTS)
    add_executable(
      test_real_data 
      src/test_real_data.c src/utils.c src/common_data.c src/io.c 
      ../dev/common.c ../src/allocators.c ../src/constructors.c 
      ../src/lapack_wrapper.c ../src/vector_algebra.c ../src/dense_algebra.c
    )

    target_link_libraries(test_real_data ${MATH_LIBRARY})
    target_link_libraries(test_real_data openblas)
    target_link_libraries(test_real_data criterion)

    if(BUILD_OPENMP_TESTS AND OpenMP_C_FOUND)
      target_link_libraries(test_real_data OpenMP::OpenMP_C)
    endif()

    if(REAL_DATA_PRINT_S)
      target_compile_definitions(test_real_data PRIVATE HODLR_REAL_DATA_PRINT_S)
    endif()
    
    add_test(NAME test_real_data COMMAND test_real_data -j1 --verbose=0)

    set_property(TEST test_real_data PROPERTY
        ENVIRONMENT "CTEST_OUTPUT_ON_FAILURE=1"
    )

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tests/data")
    file(
      CREATE_LINK 
      "${CMAKE_SOURCE_DIR}/tests/data/H" 
      "${CMAKE_BINARY_DIR}/tests/data/H" 
      SYMBOLIC
    )
  endif()

  add_custom_target(vtest COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
endif()
