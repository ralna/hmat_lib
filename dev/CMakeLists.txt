if(ASAN)
  message(STATUS "Building executable test_asan")
  add_executable(test_asan test.c ../tests/utils/io.c)
  set_target_properties(test_asan PROPERTIES
    EXCLUDE_FROM_ALL $<NOT:$<CONFIG:Debug>>
  )
  target_link_libraries(test_asan ${MATH_LIBRARY})
  target_link_libraries(test_asan openblas)
  target_link_libraries(test_asan hmat_lib)
else()
  message(STATUS "Building executable test_unsafe")
  add_executable(test_unsafe test.c ../tests/utils/io.c)
  set_target_properties(test_unsafe PROPERTIES
    EXCLUDE_FROM_ALL $<NOT:$<CONFIG:Debug>>
  )
  target_link_libraries(test_unsafe ${MATH_LIBRARY})
  target_link_libraries(test_unsafe openblas)
  target_link_libraries(test_unsafe hmat_lib)
endif()

add_executable(timings timings.c ../tests/utils/common_data.c)
target_link_libraries(timings ${MATH_LIBRARY})
target_link_libraries(timings openblas)
target_link_libraries(timings hmat_lib)
target_link_libraries(timings criterion)
target_link_libraries(timings OpenMP::OpenMP_C)

add_executable(timings2 timings2.c ../tests/utils/common_data.c)
target_link_libraries(timings2 ${MATH_LIBRARY})
target_link_libraries(timings2 openblas)
target_link_libraries(timings2 hmat_lib)
target_link_libraries(timings2 criterion)
target_link_libraries(timings2 OpenMP::OpenMP_C)


if(ASAN)
  if (CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options($<$<CONFIG:Debug>:"-fsanitize=address -fsanitize=leak -fsanitize=undefined">)
    add_link_options($<$<CONFIG:Debug>:"-fsanitize=address -fsanitize=leak -fsanitize=undefined">)
  elseif(CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    message(WARNING "Address Sanitizer is limited with AppleClang")
    add_compile_options($<$<CONFIG:Debug>:-fsanitize=address -fsanitize=undefined>)
    add_link_options($<$<CONFIG:Debug>:-fsanitize=address -fsanitize=undefined>)
  elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    message(WARNING "Address Sanitizer is limited with MSVC")
    add_compile_options($<$<CONFIG:Debug>:/fsanitize=address>)
    add_link_options($<$<CONFIG:Debug>:/fsanitize=address>)
  else()
    message(WARNING "Address Sanitizer not available for ${CMAKE_C_COMPILER_ID}")
  endif()
endif()


