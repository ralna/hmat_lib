EXTRA_FLAGS := -g

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
OUT_DIR := bin


test: test.o tree.o
	gcc ${BUILD_DIR}/test.o ${BUILD_DIR}/tree.o ${BUILD_DIR}/lapack_wrapper.o -o ${OUT_DIR}/test -llapack -lm -Wall -Wextra ${EXTRA_FLAGS}

test.o: ${SRC_DIR}/test.c
	gcc -c ${SRC_DIR}/test.c -o ${BUILD_DIR}/test.o -I ${INCLUDE_DIR} ${EXTRA_FLAGS}

tree.o: ${SRC_DIR}/tree.c lapack_wrapper.o
	gcc -c ${SRC_DIR}/tree.c -o ${BUILD_DIR}/tree.o -I ${INCLUDE_DIR} -lm -Wall -Wextra ${EXTRA_FLAGS}

lapack_wrapper.o: ${SRC_DIR}/lapack_wrapper.c
	gcc -c ${SRC_DIR}/lapack_wrapper.c -o ${BUILD_DIR}/lapack_wrapper.o -llapack -I ${INCLUDE_DIR} ${EXTRA_FLAGS}

clean:
	rm ${BUILD_DIR}/*.o ${OUT_DIR}/test

