#include <stdio.h>
#include "correctness.h"

int main(int* argc, char** argv){

    if(isCorrect(argv[1], argv[2]) != 0)
      printf("Corretto!\n");
    else
      printf("Non Corretto!\n");

}