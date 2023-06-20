#include <stdio.h>
#include <string.h>

void saveOutput(float *bodies, int nBodies, char* name){
    FILE *fp;

    fp = fopen(name, "w");

    fprintf(fp, "%d\n", nBodies);
    for(int i = 0; i<6*nBodies; i += 6){
        fprintf(fp, "Body %d: x-> %f, y-> %f, z-> %f, vx-> %f, vy-> %f, vz-> %f\n", i/6, bodies[i], bodies[i+1], bodies[i+2], bodies[i+3], bodies[i+4], bodies[i+5]);
    }
    fclose(fp);
}

void updateOutput(float *bodies, int nBodies, char* name){
    FILE *fp;

    fp = fopen(name, "a");

    fprintf(fp, "%d\n", nBodies);
    for(int i = 0; i<6*nBodies; i += 6){
        fprintf(fp, "Body %d: x-> %f, y-> %f, z-> %f, vx-> %f, vy-> %f, vz-> %f\n", i/6, bodies[i], bodies[i+1], bodies[i+2], bodies[i+3], bodies[i+4], bodies[i+5]);
    }
    fclose(fp);
}

void saveResults(float time, int nBodies, int processess, char* name, char* program, int nIters){
    FILE *fp;

    fp = fopen(name, "w");
    fprintf(fp, "Tempo di esecuzione: %0.3f\tNumero di bodies: %d\tNumero di processi: %d\tprogramma: %s\titerazioni: %d\n", time, nBodies, processess, program, nIters);
}

int isCorrect(char* f1, char* f2){
    FILE *fp1, *fp2;
    fp1 = fopen(f1, "r");
    fp2 = fopen(f2, "r");
    char riga[500], riga2[500];
    int res;
    while((res = fscanf(fp1, "%[^\n]\n", riga)) != -1){
        fscanf(fp2, "%[^\n]\n", riga2);
        if(strncmp(riga, riga2, 500) != 0)
            return 0;
    }
    return 1;
}