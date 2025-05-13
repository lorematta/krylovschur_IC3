#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Funzioni.h>  //mettere #include "Funzioni.h" per cercare prima nella directory locale del progetto

#define N 4  //dim mat A
#define P 3  //numero passi Arnoldi

void Arnoldi (double A[N][N], double v[N]){

    if (P>N){
        printf("number of steps can't be bigger than space dimension");
        exit;
    }

    
}






