test: test.o tree.o
	gcc test.o tree.o lapack_wrapper.o -o test -llapack

test.o: test.c
	gcc -c test.c -o test.o -I .

tree.o: tree.c lapack_wrapper.o
	gcc -c tree.c -o tree.o -I .

lapack_wrapper.o: lapack_wrapper.c
	gcc -c lapack_wrapper.c -o lapack_wrapper.o -llapack -I .

clean:
	rm ./*.o test

