
nbody : correctness.o nbody.o 
	mpicc correctness.o nbody.o -o nbody -lm

nbody1: correctness.o nbody1.o
	mpicc correctness.o nbody1.o -o nbody1 -lm

nbodySeq: correctness.o nbodySeq.o
	mpicc correctness.o nbodySeq.o -o nbodySeq -lm

correctness: correctness.o checkCorr.o
	mpicc correctness.o checkCorr.o -o checkCorr -lm

nbody.o :
	mpicc -c nbody.c -lm

nbody1.o :
	mpicc -c nbody1.c -lm

nbodySeq.o :
	mpicc -c nbodySeq.c -lm

correctness.o :
	mpicc -c correctness.c

checkCorr.o :
	mpicc -c checkCorr.c

clean :
	rm -f nbody.o nbody1.o nbodySeq.o correctness.o checkCorr.o nbody nbody1 nbodySeq checkCorr