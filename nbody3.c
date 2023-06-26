#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "correctness.h"
#include <string.h>

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

void myBodyForces(Body *p, float dt, int offset, int n){
  	for (int i = 0; i < n; i++) {
		int myIndex = i + offset;
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for (int j = offset; j < n + offset; j++) {
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

void otherBodyForces(Body *p, float dt, int offset, int dim, int otherOffset, int otherDim, Body *others){
    for (int i = 0; i < dim; i++) {
		int myIndex = i + offset;
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = otherOffset; j < otherDim + otherOffset; j++) {
            float dx = others[j].x - p[myIndex].x;
			float dy = others[j].y - p[myIndex].y;
			float dz = others[j].z - p[myIndex].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
			float invDist = 1.0f / sqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;
            
			Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }
		p[myIndex].vx += dt*Fx; p[myIndex].vy += dt*Fy; p[myIndex].vz += dt*Fz;
  	}
}

void calculateDisplacements(int nBodies, int processes, int myrank, int* dim, int* offset, int* receive_counts, int* displacements){
	int resto = nBodies%processes;
	*dim = nBodies/processes;
	*offset = 0;

	displacements[0] = 0;
	for(int i = 0; i < processes; i++){
		receive_counts[i] = *dim;
		if (resto > 0){
			receive_counts[i]  += 1;
			resto--;
		}
		if(i > 0)
			displacements[i] = *offset;
		*offset += receive_counts[i];
	}
	*dim = receive_counts[myrank];
	*offset = displacements[myrank];
}

void sendAndReceive(int world_size, int myrank, Body *myBodies, int displacements[], int receive_counts[], MPI_Datatype bodyDataType, MPI_Request requests[], Body *rcv){
	for(int i = 0; i<world_size; i++){
        if(i == myrank)
            MPI_Ibcast(myBodies, receive_counts[myrank], bodyDataType, i, MPI_COMM_WORLD, &requests[myrank]);
        else
            MPI_Ibcast(rcv + displacements[i], receive_counts[i], bodyDataType, i, MPI_COMM_WORLD, &requests[i]);
    }
}


void loadBuffer(Body* from, int offset, int dim, Body* to){
	int myIndex = offset;
	for(int i = 0; i < dim ; i++){
		to[i].vx = from[myIndex].vx;
		to[i].vy = from[myIndex].vy;
		to[i].vz = from[myIndex].vz;
		to[i].x = from[myIndex].x;
		to[i].y = from[myIndex].y;
		to[i].z = from[myIndex].z;
		myIndex++;
	}
}

int main(int argc, char** argv) {
	srand(0);
	MPI_Init(&argc, &argv);
	
	char* risultati = argc > 3 ? argv[3] : "output.txt";
	char* file = argc > 4 ? argv[4] : "file.txt";
	
	double start, end;
	start = MPI_Wtime();
	
	int world_size;
	int myrank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int nBodies = 30000;
	if (argc > 1) nBodies = atoi(argv[1]);
  	int dim, offset;
	int receive_counts[world_size];
	int displacements[world_size];

	calculateDisplacements(nBodies, world_size, myrank, &dim, &offset, receive_counts, displacements);

	const float dt = 0.01f; // time step
	int nIters = argc > 2 ? atoi(argv[2]) : 10; // simulation iterations

	int bytes = nBodies*sizeof(Body);
	float *buf = (float*)malloc(bytes);
	Body *p = (Body*)buf;
	Body *myBodies = malloc(dim*sizeof(Body));

	randomizeBodies(buf, 6*nBodies); // Init pos / vel data

	double totalTime = 0.0;
	MPI_Datatype bodyDataType;
	int index;
	MPI_Status status;
    MPI_Request *requests = (MPI_Request *)malloc(world_size * sizeof(MPI_Request));

	MPI_Type_contiguous(6, MPI_FLOAT, &bodyDataType);
	MPI_Type_commit(&bodyDataType);


	bodyForce(p, dt, offset, dim, nBodies); // compute interbody forces

	for (int i = 0 ; i < dim; i++) { // integrate position
		p[i + offset].x += p[i + offset].vx*dt;
		p[i + offset].y += p[i + offset].vy*dt;
		p[i + offset].z += p[i + offset].vz*dt;
	}
	
	for (int iter = 2; iter <= nIters; iter++) {
		//loadBuffer(p, offset, dim, myBodies);
		memcpy(myBodies, p+offset, sizeof(Body)*dim);
		sendAndReceive(world_size, myrank, myBodies, displacements, receive_counts, bodyDataType, requests, p);
		
		myBodyForces(p, dt, displacements[myrank], receive_counts[myrank]);
		
		for(int i = 0; i<world_size; i++){
			MPI_Waitany(world_size, requests, &index, &status);
			if(index != myrank)
				otherBodyForces(p, dt, displacements[myrank], receive_counts[myrank], displacements[index], receive_counts[index], p);
		}

		for (int i = 0 ; i < dim; i++) { // integrate position
			p[i + offset].x += p[i + offset].vx*dt;
			p[i + offset].y += p[i + offset].vy*dt;
			p[i + offset].z += p[i + offset].vz*dt;
		}
	}
	MPI_Gatherv(p + offset, dim, bodyDataType, p, receive_counts, displacements, bodyDataType, 0, MPI_COMM_WORLD);

	free(buf);
	free(myBodies);

	MPI_Type_free(&bodyDataType);
	
	end = MPI_Wtime();
	if(myrank == 0){
		printf("Tempo di esecuzione: %0.3f\tNumero di bodies: %d\tNumero di processi: %d\tprogramma: %s\titerazioni:%d\n", end-start, nBodies, world_size, argv[0], nIters);
		saveResults(end-start, nBodies, world_size, risultati, argv[0], nIters);
	}
	MPI_Finalize();
	return 0;
}
