cmake_minimum_required(VERSION 3.21)

project(hmat_lib VERSION 0.1.0 LANGUAGES C)
set(AUTHOR "Rastislav Turanyi;Hussam Al Daas")
set(CMAKE_C_STANDARD 23)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

option(ENABLE_OPENMP "Allow user to activate/deactivate OpenMP support" ON)

option(ASAN "Turns on/off Adress Sanitizer" OFF)

if(ENABLE_OPENMP)
  include(FindOpenMP)
  if(OpenMP_C_FOUND)
    message(STATUS "OpenMP for C Compiler Found, version ${OpenMP_C_VERSION_MAJOR}.${OpenMP_C_VERSION_MINOR}")
  else()
    message(ERROR_CRITICAL "No OpenMP support detected")
  endif()
endif()

find_library(MATH_LIBRARY m REQUIRED)
find_package(BLAS REQUIRED)

include(CTest)
if(BUILD_TESTING)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(criterion REQUIRED criterion)
  add_subdirectory(tests)
endif()

message(STATUS "building ${PROJECT_NAME}, version ${PROJECT_VERSION}")

include(GNUInstallDirs)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

add_subdirectory(src)


