COMPILER := gcc

$(info asan = $(asan))

ifeq ($(asan), 0)
	EXTRA_FLAGS := -g -O0
	EXE := test_unsafe
else
	EXTRA_FLAGS := -g -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined
	EXE := test_asan
endif

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
OUT_DIR := bin


test: test.o constructors.o vector_algebra.o dense_algebra.o
	${COMPILER} \
		${BUILD_DIR}/test.o \
		${BUILD_DIR}/constructors.o \
		${BUILD_DIR}/vector_algebra.o \
		${BUILD_DIR}/dense_algebra.o \
		${BUILD_DIR}/lapack_wrapper.o \
		-o ${OUT_DIR}/${EXE} \
		-llapack -lm -lopenblas \
		-Wall -Wextra \
		-fopenmp \
		${EXTRA_FLAGS}

test.o: ${SRC_DIR}/test.c
	${COMPILER} -c ${SRC_DIR}/test.c -o ${BUILD_DIR}/test.o -I ${INCLUDE_DIR} ${EXTRA_FLAGS}

constructors.o: ${SRC_DIR}/constructors.c lapack_wrapper.o
	${COMPILER} \
		-c ${SRC_DIR}/constructors.c \
		-o ${BUILD_DIR}/constructors.o \
		-I ${INCLUDE_DIR} -Wall -Wextra ${EXTRA_FLAGS} \
		-fopenmp

vector_algebra.o: ${SRC_DIR}/vector_algebra.c
	${COMPILER} \
		-c ${SRC_DIR}/vector_algebra.c \
		-o ${BUILD_DIR}/vector_algebra.o \
		-I ${INCLUDE_DIR} -Wall -Wextra ${EXTRA_FLAGS}

dense_algebra.o: ${SRC_DIR}/dense_algebra.c
	${COMPILER} \
		-c ${SRC_DIR}/dense_algebra.c \
		-o ${BUILD_DIR}/dense_algebra.o \
		-I ${INCLUDE_DIR} -Wall -Wextra ${EXTRA_FLAGS}


lapack_wrapper.o: ${SRC_DIR}/lapack_wrapper.c
	${COMPILER} -c ${SRC_DIR}/lapack_wrapper.c -o ${BUILD_DIR}/lapack_wrapper.o -llapack -I ${INCLUDE_DIR} ${EXTRA_FLAGS}

clean:
	rm ${BUILD_DIR}/*.o ${OUT_DIR}/test

