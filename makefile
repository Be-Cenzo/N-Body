
nbody1: correctness.o nbody1.o
	mpicc correctness.o nbody1.o -o nbody1 -lm

nbody2: correctness.o nbody2.o
	mpicc correctness.o nbody2.o -o nbody2 -lm

nbody3: correctness.o nbody3.o
	mpicc correctness.o nbody3.o -o nbody3 -lm

nbodySeq: correctness.o nbodySeq.o
	mpicc correctness.o nbodySeq.o -o nbodySeq -lm

correctness: correctness.o checkCorr.o
	mpicc correctness.o checkCorr.o -o checkCorr -lm

nbody1.o :
	mpicc -c nbody1.c -lm

nbody2.o :
	mpicc -c nbody2.c -lm

nbody3.o :
	mpicc -c nbody3.c -lm

nbodySeq.o :
	mpicc -c nbodySeq.c -lm

correctness.o :
	mpicc -c correctness.c

checkCorr.o :
	mpicc -c checkCorr.c

clean :
	rm -f nbody1.o nbody2.o nbody3.o nbodySeq.o correctness.o checkCorr.o nbody1 nbody2 nbody3 nbodySeq checkCorr