# if (${BUILD_SHARED_LIBS})
#   add_library(hmat_lib SHARED algebra.cxx)
#   if (WIN)
#     set (FLAGS "/Ddll_EXPORTS")
#   endif()
#
#   set_target_properties(hmat_lib PROPERTIES 
#      VERSION ${hmat_lib_VERSION}
#      SOVERSION ${hmat_lib_VERSION_MAJOR})
# else()
#    add_library(hmat_lib STATIC algebra.cxx)
# endif()
add_library(
  hmat_lib 
  allocators.c constructors.c 
  dense_algebra.c vector_algebra.c hodlr_algebra.c
  lapack_wrapper.c utils.c
)

target_link_libraries(hmat_lib ${MATH_LIBRARY})
target_link_libraries(hmat_lib openblas)

if(OpenMP_C_FOUND)
  target_link_libraries(hmat_lib OpenMP::OpenMP_C)
endif()


if(ASAN)
  message(STATUS "Building test_asan...")
  add_executable(test_asan test.c ../tests/src/io.c)
  set_target_properties(test_asan PROPERTIES
    EXCLUDE_FROM_ALL $<NOT:$<CONFIG:Debug>>
  )
  target_link_libraries(test_asan ${MATH_LIBRARY})
  target_link_libraries(test_asan openblas)
  target_link_libraries(test_asan hmat_lib)
else()
  message(STATUS "Building test_unsafe...")
  add_executable(test_unsafe test.c ../tests/src/io.c)
  set_target_properties(test_unsafe PROPERTIES
    EXCLUDE_FROM_ALL $<NOT:$<CONFIG:Debug>>
  )
  target_link_libraries(test_unsafe ${MATH_LIBRARY})
  target_link_libraries(test_unsafe openblas)
  target_link_libraries(test_unsafe hmat_lib)
endif()


if(ASAN)
  if (CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options($<$<CONFIG:Debug>:-fsanitize=address -fsanitize=leak -fsanitize=undefined>)
    add_link_options($<$<CONFIG:Debug>:-fsanitize=address -fsanitize=leak -fsanitize=undefined>)
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

