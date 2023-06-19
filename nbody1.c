#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "correctness.h"

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int offset, int n, int allBodies) {
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n; i++) {
    int myIndex = i + offset;
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < allBodies; j++) {
      float dx = p[j].x - p[myIndex].x;
      float dy = p[j].y - p[myIndex].y;
      float dz = p[j].z - p[myIndex].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }
	p[myIndex].vx += dt*Fx; p[myIndex].vy += dt*Fy; p[myIndex].vz += dt*Fz;
  }
}

void bodyPrint(Body *p){
  printf("x: %.4f, y: %.4f, z: %.4f, vx: %.2f, vy: %.2f, vz: %.2f\n", p->x, p->y, p->z, p->vx, p->vy, p->vz);
}

void calculateDisplacements(int nBodies, int processes, int myrank, int* dim, int* offset, int* receive_counts, int* displacements){
	int resto = nBodies%processes;
	*dim = nBodies/processes;
	*offset = 0;

	displacements[0] = 0;
	for(int i = 0; i < processes; i++){
		receive_counts[i] = *dim;
		if (resto > 0)
			receive_counts[i]  += 1;
		resto--;
		if(i > 0)
			displacements[i] = *offset;
		*offset += receive_counts[i];
	}
	*dim = receive_counts[myrank];
	*offset = displacements[myrank];
}

int main(int argc, char** argv) {
	srand(0);
	MPI_Init(&argc, &argv);
	
	char* risultati = argc > 2 ? argv[2] : "results.txt";
	
	double start, end;
	start = MPI_Wtime();
	
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int nBodies = 30000;
	if (argc > 1) nBodies = atoi(argv[1]);
  	int dim, offset;
	int receive_counts[world_size];
	int displacements[world_size];

	calculateDisplacements(nBodies, world_size, myrank, &dim, &offset, receive_counts, displacements);

	const float dt = 0.01f; // time step
	const int nIters = 10;  // simulation iterations

	int bytes = nBodies*sizeof(Body);
	float *buf = (float*)malloc(bytes);
	Body *p = (Body*)buf;

	if(myrank == 0)
		randomizeBodies(buf, 6*nBodies); // Init pos / vel data

	double totalTime = 0.0;
	MPI_Datatype bodyDataType;

	MPI_Type_contiguous(6, MPI_FLOAT, &bodyDataType);
	MPI_Type_commit(&bodyDataType);
	
	for (int iter = 1; iter <= nIters; iter++) {
		MPI_Bcast(p, nBodies, bodyDataType, 0, MPI_COMM_WORLD);

		bodyForce(p, dt, offset, dim, nBodies); // compute interbody forces

		for (int i = 0 ; i < dim; i++) { // integrate position
			p[i + offset].x += p[i + offset].vx*dt;
			p[i + offset].y += p[i + offset].vy*dt;
			p[i + offset].z += p[i + offset].vz*dt;
		}
		MPI_Gatherv(p + offset, dim, bodyDataType, p, receive_counts, displacements, bodyDataType, 0, MPI_COMM_WORLD);

	}

	free(buf);

	MPI_Type_free(&bodyDataType);
	
	end = MPI_Wtime();
	if(myrank == 0){
		printf("Tempo di esecuzione: %0.3f\tNumero di bodies: %d\tNumero di processi: %d\tprogramma: %s\n", end-start, nBodies, world_size, argv[0]);
		saveResults(end-start, nBodies, world_size, risultati, argv[0]);
	}
		
	MPI_Finalize();
	return 0;
}
