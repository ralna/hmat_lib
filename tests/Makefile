SRC_DIR := ../src
TESTS_DIR := src
INCLUDE_DIR := ../include
BUILD_DIR := build
OUT_DIR := ../bin


test: test_tree.o
	gcc ${BUILD_DIR}/test_tree.o ${BUILD_DIR}/lapack_wrapper.o -o ${OUT_DIR}/test_tree -lopenblas -lm -Wall -Wextra -lcriterion -g

test_tree.o: ${TESTS_DIR}/test_tree.c lapack_wrapper.o
	gcc -c ${TESTS_DIR}/test_tree.c -o ${BUILD_DIR}/test_tree.o -I ${INCLUDE_DIR} -g

tree.o: ${SRC_DIR}/tree.c lapack_wrapper.o
	gcc -c ${SRC_DIR}/tree.c -o ${BUILD_DIR}/tree.o -I ${INCLUDE_DIR} -lm -Wall -Wextra -g

lapack_wrapper.o: ${SRC_DIR}/lapack_wrapper.c
	gcc -c ${SRC_DIR}/lapack_wrapper.c -o ${BUILD_DIR}/lapack_wrapper.o -lopenblas -I ${INCLUDE_DIR} -g

clean:
	rm ${BUILD_DIR}/*.o ${OUT_DIR}/test_tree

