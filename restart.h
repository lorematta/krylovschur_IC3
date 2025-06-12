#ifndef RESTART_H
#define RESTART_H

#include <slepceps.h>

PetscErrorCode SaveRestartData(Vec *V, PetscInt k);
PetscErrorCode LoadRestartData(Vec **V_restart, PetscInt *k_restart, Vec v0);
//PetscErrorCode MatMult_MarkovModel(Mat A, Vec x, Vec y);
PetscErrorCode MatMarkovModel(PetscInt m,Mat A);



#endif // RESTART_H
