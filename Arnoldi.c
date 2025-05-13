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
        double H [i][i];
        double H_new [i+1][i+1];
        matvec(A, v, r);  // r = A*v_k  dim N
        mattrasp(N, V);   // V^T  
        matvec(V, r, h);  // h = V^T *r  dim P+1
        mattrasp(N, V);   // V  
        double y [P+1];
        matvec(V,h,y); // r = r - V*h  dim N

        for (int j = 0; i<N ; i++){
            r[j] -= y[j];
        }

        b = norma(r);
        int flag1 = 0;
        int flag2 = 0; 
        for (int j = 0; j<i; j++){
            for (int k = 0; k<j; k++){
                H_new[j][k] =  H[j][k];
                flag1 = k;    
            }
            H_new[j][flag1+1] = h[j];
            flag2 = j;
        }
        H_new[flag2+1][0] = b;
        
        
    }
    
    
}





