SRC_DIR := ../src
TESTS_DIR := unit_tests

HOME_DIR := ./
UTIL_DIR := utils

INCLUDE_DIR := ../include/hmat_lib
INTERNAL_INCLUDE_DIR := ../include/internal

DEV_DIR := ../dev
BUILD_DIR := build
OUT_DIR := ../bin

EXTRA_FLAGS := -g #-fsanitize=address

WARNINGS := -Wall -Wextra -Wno-unknown-pragmas

COMPILER := gcc

REAL_DATA_PRINT_S := #-DHODLR_REAL_DATA_PRINT_S


unit_tests: test_allocators.o test_tree_construct.o test_vector_algebra.o test_dense_algebra.o test_hodlr_algebra.o lapack_wrapper.o utils.o common_data.o main_utils.o hodlr.o
	gcc \
		${BUILD_DIR}/test_allocators.o \
		${BUILD_DIR}/test_vector_algebra.o \
		${BUILD_DIR}/test_tree_construct.o \
		${BUILD_DIR}/test_dense_algebra.o \
		${BUILD_DIR}/test_hodlr_algebra.o \
		${BUILD_DIR}/lapack_wrapper.o \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/hodlr.o \
		${BUILD_DIR}/main_utils.o \
		${BUILD_DIR}/common_data.o \
		-o ${OUT_DIR}/unit_test \
		-lopenblas -lm -lcriterion \
		${EXTRA_FLAGS} -fopenmp


test_hodlr_algebra.o: ${TESTS_DIR}/test_hodlr_hodlr_algebra.c
	gcc \
		-c ${TESTS_DIR}/test_hodlr_hodlr_algebra.c \
		-o ${BUILD_DIR}/test_hodlr_algebra.o \
		-lopenblas -lm -lcriterion \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${WARNINGS} \
		${EXTRA_FLAGS}

test_dense_algebra.o: ${TESTS_DIR}/test_hodlr_dense_algebra.c lapack_wrapper.o utils.o common_data.o
	gcc \
		-c ${TESTS_DIR}/test_hodlr_dense_algebra.c \
		-o ${BUILD_DIR}/test_dense_algebra.o \
		-lopenblas -lm -lcriterion \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${WARNINGS} \
		${EXTRA_FLAGS}

test_vector_algebra.o: ${TESTS_DIR}/test_vector_algebra.c lapack_wrapper.o utils.o common_data.o 
	gcc \
		-c ${TESTS_DIR}/test_vector_algebra.c \
		-o ${BUILD_DIR}/test_vector_algebra.o \
		-lopenblas -lm -lcriterion -lmimick \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}

test_tree_construct.o: ${TESTS_DIR}/test_tree_construct.c
	gcc \
		-c ${TESTS_DIR}/test_tree_construct.c \
		-o ${BUILD_DIR}/test_tree_construct.o \
		-lopenblas \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${WARNINGS} \
		${EXTRA_FLAGS}

test_allocators.o: ${TESTS_DIR}/test_allocators.c
	gcc \
		-c ${TESTS_DIR}/test_allocators.c \
		-o ${BUILD_DIR}/test_allocators.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}



test_real_data: test_real_data.o utils.o common_data.o io.o allocators.o constructors_omp.o lapack_wrapper.o vector_algebra.o dense_algebra.o common.o hodlr_algebra.o main_utils.o
	gcc \
		${BUILD_DIR}/test_real_data.o \
		${BUILD_DIR}/allocators.o \
		${BUILD_DIR}/constructors_omp.o \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/hodlr.o \
		${BUILD_DIR}/main_utils.o \
		${BUILD_DIR}/common_data.o \
		${BUILD_DIR}/io.o \
		${BUILD_DIR}/common.o \
		${BUILD_DIR}/lapack_wrapper.o \
		${BUILD_DIR}/vector_algebra.o \
		${BUILD_DIR}/dense_algebra.o \
		${BUILD_DIR}/hodlr_algebra.o \
		-o ${OUT_DIR}/test_real_data \
		-lopenblas -lm -lcriterion \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${WARNINGS} ${EXTRA_FLAGS} \
		-fopenmp \
		${REAL_DATA_PRINT_S}

test_real_data.o: ${HOME_DIR}/test_real_data.c
	gcc \
	  -c ${HOME_DIR}/test_real_data.c \
		-o ${BUILD_DIR}/test_real_data.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS} ${REAL_DATA_PRINT_S} -fopenmp



allocators.o: ${SRC_DIR}/allocators.c
	${COMPILER} \
		-c ${SRC_DIR}/allocators.c \
		-o ${BUILD_DIR}/allocators.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} \
		${WARNINGS} ${EXTRA_FLAGS}

constructors.o: ${SRC_DIR}/constructors.c
	${COMPILER} \
		-c ${SRC_DIR}/constructors.c \
		-o ${BUILD_DIR}/constructors.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} \
		${WARNINGS} ${EXTRA_FLAGS} 

constructors_omp.o: ${SRC_DIR}/constructors.c
	${COMPILER} \
		-c ${SRC_DIR}/constructors.c \
		-o ${BUILD_DIR}/constructors_omp.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} \
		${WARNINGS} ${EXTRA_FLAGS} \
		-fopenmp

lapack_wrapper.o: ${SRC_DIR}/lapack_wrapper.c
	gcc \
		-c ${SRC_DIR}/lapack_wrapper.c \
		-o ${BUILD_DIR}/lapack_wrapper.o \
		-lopenblas \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} \
		${EXTRA_FLAGS}

vector_algebra.o: ${SRC_DIR}/vector_algebra.c
	${COMPILER} \
		-c ${SRC_DIR}/vector_algebra.c \
		-o ${BUILD_DIR}/vector_algebra.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} \
		${WARNINGS} ${EXTRA_FLAGS}

dense_algebra.o: ${SRC_DIR}/dense_algebra.c
	${COMPILER} \
		-c ${SRC_DIR}/dense_algebra.c \
		-o ${BUILD_DIR}/dense_algebra.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} \
		${WARNINGS} ${EXTRA_FLAGS}

hodlr_algebra.o: ${SRC_DIR}/hodlr_algebra.c
	${COMPILER} \
		-c ${SRC_DIR}/hodlr_algebra.c \
		-o ${BUILD_DIR}/hodlr_algebra.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} \
		${WARNINGS} ${EXTRA_FLAGS}



utils.o: ${UTIL_DIR}/utils.c
	gcc \
		-c ${UTIL_DIR}/utils.c \
		-o ${BUILD_DIR}/utils.o \
		-lm -lcriterion \
		-I ${UTIL_DIR} \
		${EXTRA_FLAGS}

hodlr.o: ${UTIL_DIR}/hodlr.c
	gcc \
		-c ${UTIL_DIR}/hodlr.c \
		-o ${BUILD_DIR}/hodlr.o \
		-lm -lcriterion \
		-I ${UTIL_DIR} \
		${EXTRA_FLAGS}

common_data.o: ${UTIL_DIR}/common_data.c
	gcc \
		-c ${UTIL_DIR}/common_data.c \
		-o ${BUILD_DIR}/common_data.o \
		-lcriterion \
		-I ${UTIL_DIR} \
		${EXTRA_FLAGS}

io.o: ${UTIL_DIR}/io.c
	gcc \
		-c ${UTIL_DIR}/io.c \
		-o ${BUILD_DIR}/io.o \
		-lcriterion \
		-I ${UTIL_DIR} \
		${WARNINGS} ${EXTRA_FLAGS} -D_TEST_HODLR

common.o: ${DEV_DIR}/common.c
	gcc \
		-c ${DEV_DIR}/common.c \
		-o ${BUILD_DIR}/common.o \
		-I ${DEV_DIR} \
		${WARNINGS} ${EXTRA_FLAGS} -D_TEST_HODLR

main_utils.o: ${SRC_DIR}/utils.c
	gcc \
		-c ${SRC_DIR}/utils.c \
		-o ${BUILD_DIR}/main_utils.o \
		-I ${INCLUDE_DIR} -I ${INTERNAL_INCLUDE_DIR} \
		${EXTRA_FLAGS}


clean:
	rm ${BUILD_DIR}/*.o ${OUT_DIR}/test_tree

