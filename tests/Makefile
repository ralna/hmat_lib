SRC_DIR := ../src
TESTS_DIR := src
UTIL_INCLUDE_DIR := include
INCLUDE_DIR := ../include
BUILD_DIR := build
OUT_DIR := ../bin

EXTRA_FLAGS := -g #-fsanitize=address

WARNINGS := -Wall -Wextra -Wno-unknown-pragmas

COMPILER := gcc


unit_tests: test_allocators.o test_tree_construct.o test_vector_algebra.o test_dense_algebra.o lapack_wrapper.o utils.o common_data.o
	gcc \
		${BUILD_DIR}/test_allocators.o \
		${BUILD_DIR}/test_vector_algebra.o \
		${BUILD_DIR}/test_tree_construct.o \
		${BUILD_DIR}/test_dense_algebra.o \
		${BUILD_DIR}/lapack_wrapper.o \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/common_data.o \
		-o ${OUT_DIR}/unit_test \
		-lopenblas -lm -lcriterion \
		${EXTRA_FLAGS} -fopenmp


test_dense_algebra: test_dense_algebra.o allocators.o constructors.o
	gcc \
		${BUILD_DIR}/test_dense_algebra.o \
		${BUILD_DIR}/lapack_wrapper.o \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/common_data.o \
		${BUILD_DIR}/allocators.o \
		${BUILD_DIR}/constructors.o \
		-o ${OUT_DIR}/test_dense_algebra \
		-lopenblas -lm -lcriterion \
		${EXTRA_FLAGS}

test_dense_algebra.o: ${TESTS_DIR}/test_hodlr_dense_algebra.c lapack_wrapper.o utils.o common_data.o
	gcc \
		-c ${TESTS_DIR}/test_hodlr_dense_algebra.c \
		-o ${BUILD_DIR}/test_dense_algebra.o \
		-lopenblas -lm -lcriterion \
		-I ${INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${WARNINGS} \
		${EXTRA_FLAGS}


test_vector_algebra: test_vector_algebra.o
	gcc \
		${BUILD_DIR}/test_vector_algebra.o \
		${BUILD_DIR}/lapack_wrapper.o \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/common_data.o \
		-o ${OUT_DIR}/test_vector_algebra \
		-lopenblas -lm -lcriterion \
		-Wall -Wextra \
		${EXTRA_FLAGS}

test_vector_algebra.o: ${TESTS_DIR}/test_vector_algebra.c lapack_wrapper.o utils.o common_data.o 
	gcc \
		-c ${TESTS_DIR}/test_vector_algebra.c \
		-o ${BUILD_DIR}/test_vector_algebra.o \
		-lopenblas -lm -lcriterion -lmimick \
		-I ${INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}


test_tree_construct: test_tree_construct.o allocators.o lapack_wrapper.o utils.o common_data.o 
	gcc \
		${BUILD_DIR}/test_tree_construct.o \
		${BUILD_DIR}/allocators.o \
		${BUILD_DIR}/lapack_wrapper.o  \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/common_data.o \
		-o ${OUT_DIR}/test_tree_construct \
		-lopenblas -lm -lcriterion \
		-Wall -Wextra \
		${EXTRA_FLAGS}

test_tree_construct.o: ${TESTS_DIR}/test_tree_construct.c
	gcc \
		-c ${TESTS_DIR}/test_tree_construct.c \
		-o ${BUILD_DIR}/test_tree_construct.o \
		-lopenblas \
		-I ${INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${WARNINGS} \
		${EXTRA_FLAGS}

test_allocators: test_allocators.o utils.o
	gcc \
		${BUILD_DIR}/test_allocators.o \
		${BUILD_DIR}/utils.o \
		-o ${OUT_DIR}/test_allocators \
		-lopenblas -lm -lcriterion \
		-Wall -Wextra \
		${EXTRA_FLAGS}

test_allocators.o: ${TESTS_DIR}/test_allocators.c
	gcc \
		-c ${TESTS_DIR}/test_allocators.c \
		-o ${BUILD_DIR}/test_allocators.o \
		-I ${INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}

test_real_data: test_real_data.o utils.o common_data.o io.o allocators.o constructors_omp.o lapack_wrapper.o vector_algebra.o dense_algebra.o
	gcc \
		${BUILD_DIR}/test_real_data.o \
		${BUILD_DIR}/allocators.o \
		${BUILD_DIR}/constructors_omp.o \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/common_data.o \
		${BUILD_DIR}/io.o \
		${BUILD_DIR}/lapack_wrapper.o \
		${BUILD_DIR}/vector_algebra.o \
		${BUILD_DIR}/dense_algebra.o \
		-o ${OUT_DIR}/test_real_data \
		-lopenblas -lm -lcriterion \
		-I ${INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${WARNINGS} ${EXTRA_FLAGS} \
		-fopenmp

test_real_data.o: ${TESTS_DIR}/test_real_data.c
	gcc \
		-c ${TESTS_DIR}/test_real_data.c \
		-o ${BUILD_DIR}/test_real_data.o \
		-I ${INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}



tree.o: ${SRC_DIR}/tree.c lapack_wrapper.o
	gcc -c ${SRC_DIR}/tree.c -o ${BUILD_DIR}/tree.o -I ${INCLUDE_DIR} -lm -Wall -Wextra -g

allocators.o: ${SRC_DIR}/allocators.c
	${COMPILER} \
		-c ${SRC_DIR}/allocators.c \
		-o ${BUILD_DIR}/allocators.o \
		-I ${INCLUDE_DIR} ${WARNINGS} ${EXTRA_FLAGS}

constructors.o: ${SRC_DIR}/constructors.c
	${COMPILER} \
		-c ${SRC_DIR}/constructors.c \
		-o ${BUILD_DIR}/constructors.o \
		-I ${INCLUDE_DIR} ${WARNINGS} ${EXTRA_FLAGS} 

constructors_omp.o: ${SRC_DIR}/constructors.c
	${COMPILER} \
		-c ${SRC_DIR}/constructors.c \
		-o ${BUILD_DIR}/constructors_omp.o \
		-I ${INCLUDE_DIR} ${WARNINGS} ${EXTRA_FLAGS} \
		-fopenmp

lapack_wrapper.o: ${SRC_DIR}/lapack_wrapper.c
	gcc \
		-c ${SRC_DIR}/lapack_wrapper.c \
		-o ${BUILD_DIR}/lapack_wrapper.o \
		-lopenblas \
		-I ${INCLUDE_DIR} \
		${EXTRA_FLAGS}

vector_algebra.o: ${SRC_DIR}/vector_algebra.c
	${COMPILER} \
		-c ${SRC_DIR}/vector_algebra.c \
		-o ${BUILD_DIR}/vector_algebra.o \
		-I ${INCLUDE_DIR} ${WARNINGS} ${EXTRA_FLAGS}

dense_algebra.o: ${SRC_DIR}/dense_algebra.c
	${COMPILER} \
		-c ${SRC_DIR}/dense_algebra.c \
		-o ${BUILD_DIR}/dense_algebra.o \
		-I ${INCLUDE_DIR} ${WARNINGS} ${EXTRA_FLAGS}



utils.o: ${TESTS_DIR}/utils.c
	gcc \
		-c ${TESTS_DIR}/utils.c \
		-o ${BUILD_DIR}/utils.o \
		-lm -lcriterion \
		-I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}

common_data.o: ${TESTS_DIR}/common_data.c
	gcc \
		-c ${TESTS_DIR}/common_data.c \
		-o ${BUILD_DIR}/common_data.o \
		-lcriterion \
		-I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}

io.o: ${TESTS_DIR}/io.c
	gcc \
		-c ${TESTS_DIR}/io.c \
		-o ${BUILD_DIR}/io.o \
		-lcriterion \
		-I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}



clean:
	rm ${BUILD_DIR}/*.o ${OUT_DIR}/test_tree

