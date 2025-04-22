COMPILER := gcc

EXTRA_FLAGS := -g -O0 #-fsanitize=address -fsanitize=leak -fsanitize=undefined

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
OUT_DIR := bin


test: test.o tree.o
	${COMPILER} \
		${BUILD_DIR}/test.o \
		${BUILD_DIR}/tree.o \
		${BUILD_DIR}/lapack_wrapper.o \
		-o ${OUT_DIR}/test \
		-llapack -lm -lopenblas \
		-Wall -Wextra \
		${EXTRA_FLAGS}

test.o: ${SRC_DIR}/test.c
	${COMPILER} -c ${SRC_DIR}/test.c -o ${BUILD_DIR}/test.o -I ${INCLUDE_DIR} ${EXTRA_FLAGS}

tree.o: ${SRC_DIR}/tree.c lapack_wrapper.o
	${COMPILER} -c ${SRC_DIR}/tree.c -o ${BUILD_DIR}/tree.o -I ${INCLUDE_DIR} -lm -Wall -Wextra ${EXTRA_FLAGS}

lapack_wrapper.o: ${SRC_DIR}/lapack_wrapper.c
	${COMPILER} -c ${SRC_DIR}/lapack_wrapper.c -o ${BUILD_DIR}/lapack_wrapper.o -llapack -I ${INCLUDE_DIR} ${EXTRA_FLAGS}

clean:
	rm ${BUILD_DIR}/*.o ${OUT_DIR}/test

