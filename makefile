
nbody1: correctness.o nbody1.o
	mpicc correctness.o nbody1.o -o nbody1 -lm

nbody2: correctness.o nbody2.o
	mpicc correctness.o nbody2.o -o nbody2 -lm

nbodySeq: correctness.o nbodySeq.o
	mpicc correctness.o nbodySeq.o -o nbodySeq -lm

nbody1.o :
	mpicc -c nbody1.c -lm

nbody2.o :
	mpicc -c nbody2.c -lm

nbodySeq.o :
	mpicc -c nbodySeq.c -lm

correctness.o :
	mpicc -c correctness.c

clean :
	rm -f nbody1.o nbody2.o nbodySeq.o correctness.o nbody1 nbody2 nbodySeq 