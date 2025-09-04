message(STATUS "Building documentation")

set(DOCS_DIR "${CMAKE_SOURCE_DIR}/docs/source")
set(BUILD_DOCS_DIR "${CMAKE_BINARY_DIR}/docs")

find_package(Python3 3 COMPONENTS Interpreter)
if(NOT Python3_FOUND)
  message(WARNING "Python3 not found - aborting documentation build")
  return()
endif()

# Check Sphinx existence
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m sphinx --version
    RESULT_VARIABLE sphinx_result
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT sphinx_result EQUAL 0)
  message(WARNING "Sphinx not found - aborting documentation build")
  return()
endif()

# Check Hawkmoth existence
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m hawkmoth --version
    RESULT_VARIABLE sphinx_result
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT sphinx_result EQUAL 0)
  message(WARNING "Hawkmoth not found - aborting documentation build")
  return()
endif()


add_custom_target(doc ALL
  COMMAND ${Python3_EXECUTABLE} -m sphinx -b html ${DOCS_DIR} ${BUILD_DOCS_DIR}
  COMMENT "Building HTML documentation with Sphinx"
)

add_dependencies(hmat_lib doc)

