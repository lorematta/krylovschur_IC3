#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4  //dim mat A
#define P 3  //numero passi Arnoldi



void matvec (double A[N][N], double x[N], double y[N]){    //funzione per eseguire il prodotto matrice-vettore

    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            y[i] += A[i][j]*x[j]; 
        }
    }

}

double norma (double *v){           // RB:  * indica dereferenziazione, si estrae il valore dall indirizzo di memoria associato a v 
    return sqrt (prodscal(v,v));    //      con & si indica l'indirizzon della varabile puntata
}


double prodscal (double *v, double *w){
    double somma = 0.0;
    for (int i = 0; i < N; i++){
        somma += v[i]*w[i];
    }
    return somma;
}

void mattrasp (int dim, double A[dim][dim]){
    for (int i=0; i<dim; i++){
        for (int j=i+1; j<dim; j++){
            double temp = A[i][j];
            A[i][j] = A[j][i];
            A[j][i] = temp;
        }
    }
}
#endif


