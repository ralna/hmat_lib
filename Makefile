EXTRA_FLAGS := -g

test: test.o tree.o
	gcc test.o tree.o lapack_wrapper.o -o test -llapack -lm -Wall -Wextra ${EXTRA_FLAGS}

test.o: test.c
	gcc -c test.c -o test.o -I . ${EXTRA_FLAGS}

tree.o: tree.c lapack_wrapper.o
	gcc -c tree.c -o tree.o -I . -lm -Wall -Wextra ${EXTRA_FLAGS}

lapack_wrapper.o: lapack_wrapper.c
	gcc -c lapack_wrapper.c -o lapack_wrapper.o -llapack -I . ${EXTRA_FLAGS}

clean:
	rm ./*.o test

