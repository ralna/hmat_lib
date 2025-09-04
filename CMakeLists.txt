cmake_minimum_required(VERSION 3.21)

##### PROJECT INFORMATION #####
project(hmat_lib VERSION 0.1.0 LANGUAGES C)
set(AUTHOR "Rastislav Turanyi;Hussam Al Daas")
set(CMAKE_C_STANDARD 23)

##### OPTIONS #####
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ENABLE_OPENMP "Allow user to activate/deactivate OpenMP support" ON)
option(ASAN "Turns on/off Adress Sanitizer" OFF)

# Documentation options:
option(BUILD_DOCS "Build documentation" ON)

# Testing options:
include(CMakeDependentOption)
include(CTest)

cmake_dependent_option(
  BUILD_OPENMP_TESTS 
  "In addition to normal tests, also builds and runs tests with OpenMP" 
  ON
  BUILD_TESTING
  OFF
)
cmake_dependent_option(
  BUILD_REAL_DATA_TESTS 
  "\
Builds and runs tests that use (large) real data \
(NOTE: corresponding data files not included in the repository)\
" 
  OFF 
  BUILD_TESTING
  OFF
)
cmake_dependent_option(
  REAL_DATA_PRINT_S 
  "Causes real data tests to print the values of `s` of each node." 
  OFF
  BUILD_REAL_DATA_TESTS
  OFF
)

##### LOAD MODULES #####
if(ENABLE_OPENMP)
  include(FindOpenMP)
  if(OpenMP_C_FOUND)
    message(STATUS "OpenMP for C Compiler Found, version ${OpenMP_C_VERSION_MAJOR}.${OpenMP_C_VERSION_MINOR}")
  else()
    message(ERROR_CRITICAL "No OpenMP support detected")
  endif()
else()
  if(BUILD_OPENMP_TESTS)
    message(WARNING "Testing with OpenMP but building without")
    include(FindOpenMP)
    if(NOT OpenMP_C_FOUND)
      message(WARNING "No OpenMP support detected")
    endif()
  endif()
endif()

find_library(MATH_LIBRARY m REQUIRED)
find_package(BLAS REQUIRED)

if(BUILD_TESTING)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(criterion REQUIRED criterion)
  add_subdirectory(tests)
endif()

message(STATUS "building ${PROJECT_NAME}, version ${PROJECT_VERSION}")

include(GNUInstallDirs)

##### PATHS #####
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

##### MAIN #####
add_subdirectory(src)
add_subdirectory(dev)

if(BUILD_DOCS)
  add_subdirectory(docs)
endif()


