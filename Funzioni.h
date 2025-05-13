#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4
#define P 3

double prodscal(double *v, double *w);
void matvec(double A[N][N], double x[N], double y[N]);
double norma(double *v);
void mattrasp(int dim, double A[dim][dim]);

#endif
