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
  message(STATUS "Building library with ASAN ON")
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
else()
  message(STATUS "Building library with ASAN OFF")
endif()

