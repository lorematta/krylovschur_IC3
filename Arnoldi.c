#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Funzioni.h" // in xcode non serve mettere #include, rispetto a vscode usa "" e non <>

#define N 4  //dim mat A
#define P 3  //numero passi Arnoldi

void Arnoldi (double A[N][N], double v[N]){


    
    if (P>N){
        printf("number of steps can't be bigger than space dimension"); // fprintf stamperebbe su *FILE generico e non solo su stdoutput
        exit(EXIT_FAILURE);
    }else if (norma(v) != 1){
        printf("input vector is not a unitary vector");
        exit(EXIT_FAILURE);
    }
    
    double H[P+1][P];
    double V[N][P+1];
    double r[N];
    double h[N];
    
    for(int i=0; i<N; i++){
        for (int j = 0; j<N; j++){
            V[j][i] = v[i];
        }
    }
    
    for (int i = 0; i < P-2; i++){
        matvec(A, v, r);
        matvec(V, r, h);
        
    }
    
    

    
}





