SRC_DIR := ../src
TESTS_DIR := src
UTIL_INCLUDE_DIR := include
INCLUDE_DIR := ../include
BUILD_DIR := build
OUT_DIR := ../bin

EXTRA_FLAGS := -g


test: test_tree_construct.o
	gcc \
		${BUILD_DIR}/test_tree_construct.o \
		${BUILD_DIR}/lapack_wrapper.o  \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/common_data.o \
		-o ${OUT_DIR}/test_tree \
		-lopenblas -lm -lcriterion \
		-Wall -Wextra \
		${EXTRA_FLAGS}

test_tree_construct.o: ${TESTS_DIR}/test_tree_construct.c lapack_wrapper.o utils.o common_data.o
	gcc \
		-c ${TESTS_DIR}/test_tree_construct.c \
		-o ${BUILD_DIR}/test_tree_construct.o \
		-I ${INCLUDE_DIR} -I ${UTIL_INCLUDE_DIR} \
		${EXTRA_FLAGS}



tree.o: ${SRC_DIR}/tree.c lapack_wrapper.o
	gcc -c ${SRC_DIR}/tree.c -o ${BUILD_DIR}/tree.o -I ${INCLUDE_DIR} -lm -Wall -Wextra -g

lapack_wrapper.o: ${SRC_DIR}/lapack_wrapper.c
	gcc \
		-c ${SRC_DIR}/lapack_wrapper.c \
		-o ${BUILD_DIR}/lapack_wrapper.o \
		-lopenblas \
		-I ${INCLUDE_DIR} \
		${EXTRA_FLAGS}



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



clean:
	rm ${BUILD_DIR}/*.o ${OUT_DIR}/test_tree

