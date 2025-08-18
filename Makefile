COMPILER := gcc

$(info asan = $(asan))

ifeq ($(asan), 0)
	EXTRA_FLAGS := -g -O0
	EXE := test_unsafe
else
	EXTRA_FLAGS := -g -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined
	EXE := test_asan
endif

OMP := #-fopenmp

WARNINGS := -Wall -Wextra -Wno-unknown-pragmas

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
OUT_DIR := bin

OTHER_DIR := ignore


test: test.o allocators.o constructors.o vector_algebra.o dense_algebra.o io.o utils.o
	${COMPILER} \
		${BUILD_DIR}/test.o \
		${BUILD_DIR}/allocators.o \
		${BUILD_DIR}/constructors.o \
		${BUILD_DIR}/vector_algebra.o \
		${BUILD_DIR}/dense_algebra.o \
		${BUILD_DIR}/utils.o \
		${BUILD_DIR}/lapack_wrapper.o \
		${BUILD_DIR}/io.o \
		-o ${OUT_DIR}/${EXE} \
		-llapack -lm -lopenblas \
		${WARNINGS} \
		${OMP} \
		${EXTRA_FLAGS}

test.o: ${SRC_DIR}/test.c
	${COMPILER} -c ${SRC_DIR}/test.c -o ${BUILD_DIR}/test.o -I ${INCLUDE_DIR} ${EXTRA_FLAGS}

compare: compare.o lapack_wrapper.o
	${COMPILER} \
		${BUILD_DIR}/compare.o \
		${BUILD_DIR}/lapack_wrapper.o \
		-o ${OUT_DIR}/compare \
		-llapack -lm -lopenblas \
		${WARNINGS} \
		${EXTRA_FLAGS}

compare.o: ${OTHER_DIR}/compare_looping.c lapack_wrapper.o
	${COMPILER} \
		-c ${OTHER_DIR}/compare_looping.c \
		-o ${BUILD_DIR}/compare.o \
		-I ${INCLUDE_DIR} ${WARNINGS} ${EXTRA_FLAGS} \


allocators.o: ${SRC_DIR}/allocators.c
	${COMPILER} \
		-c ${SRC_DIR}/allocators.c \
		-o ${BUILD_DIR}/allocators.o \
		-I ${INCLUDE_DIR} ${WARNINGS} ${EXTRA_FLAGS} \

constructors.o: ${SRC_DIR}/constructors.c lapack_wrapper.o
	${COMPILER} \
		-c ${SRC_DIR}/constructors.c \
		-o ${BUILD_DIR}/constructors.o \
		-I ${INCLUDE_DIR} ${WARNINGS} ${EXTRA_FLAGS} \
		${OMP}

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

utils.o: ${SRC_DIR}/utils.c
	gcc \
		-c ${SRC_DIR}/utils.c \
		-o ${BUILD_DIR}/utils.o \
		-I ${INCLUDE_DIR} \
		${EXTRA_FLAGS} ${WARNINGS}



lapack_wrapper.o: ${SRC_DIR}/lapack_wrapper.c
	${COMPILER} \
		-c ${SRC_DIR}/lapack_wrapper.c \
		-o ${BUILD_DIR}/lapack_wrapper.o \
		-llapack -I ${INCLUDE_DIR} ${EXTRA_FLAGS}

io.o: tests/src/io.c
	${COMPILER} -c tests/src/io.c -o ${BUILD_DIR}/io.o -I tests/include ${EXTRA_FLAGS}


clean:
	rm ${BUILD_DIR}/*.o ${OUT_DIR}/test

