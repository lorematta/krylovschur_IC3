#include "restart.h"


PetscErrorCode SaveRestartData(Vec *V, PetscInt k) {
    PetscViewer viewer;
    
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "restart_data.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerBinaryWrite(viewer, &k, 1, PETSC_INT));

    for (PetscInt i = 0; i < k; ++i) {
        PetscCall(VecView(V[i], viewer));  // Save each vector
    }

    PetscCall(PetscViewerDestroy(&viewer));
    
    return 0;    
}

PetscErrorCode LoadRestartData(Vec **V_restart, PetscInt *k_restart, Vec v0) {
    PetscViewer viewer;
    
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "restart_data.dat", FILE_MODE_READ, &viewer));

    // Read number of vectors
    PetscCall(PetscViewerBinaryRead(viewer, k_restart, 1, NULL, PETSC_INT));

    PetscCall(PetscMalloc1(*k_restart, V_restart));  // Allocate memory for vectors

    for (PetscInt i = 0; i < *k_restart; ++i) {
        PetscCall(VecDuplicate(v0, &((*V_restart)[i])));  // Initialize vector
        PetscCall(VecLoad((*V_restart)[i], viewer));  // Load data into vector
    }

    PetscCall(PetscViewerDestroy(&viewer));
    return 0;  // Successfully loaded restart data

}

PetscErrorCode MatMult_MarkovModel(Mat A, Vec x, Vec y) {
    void *contex;
    const PetscScalar *px;
    PetscScalar       *py;

    PetscCall(MatShellGetContext(A, &contex));
    Mat B = *(Mat*)contex;
   
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged? Matmult\n"));

    PetscCall(VecGetArrayRead(x, &px));
    PetscCall(VecGetArray(y, &py));

    MatMult(B, x, y);


    PetscCall(VecRestoreArrayRead(x, &px));
    PetscCall(VecRestoreArray(y, &py));

}



PetscErrorCode MatMarkovModel(PetscInt m,Mat A) {
  
    const PetscReal cst = 0.5/(PetscReal)(m-1);
    PetscReal       pd,pu;
    PetscInt        Istart,Iend,i,j,jmax,ix=0;

    PetscFunctionBeginUser;
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (i=1;i<=m;i++) {
        jmax = m-i+1;
        for (j=1;j<=jmax;j++) {
            ix = ix + 1;
            if (ix-1<Istart || ix>Iend) continue;  /* compute only owned rows */
            if (j!=jmax) {
                pd = cst*(PetscReal)(i+j-1);
                /* north */
                if (i==1) PetscCall(MatSetValue(A,ix-1,ix,2*pd,INSERT_VALUES));
                else PetscCall(MatSetValue(A,ix-1,ix,pd,INSERT_VALUES));
                /* east */
                if (j==1) PetscCall(MatSetValue(A,ix-1,ix+jmax-1,2*pd,INSERT_VALUES));
                else PetscCall(MatSetValue(A,ix-1,ix+jmax-1,pd,INSERT_VALUES));
            }
            /* south */
            pu = 0.5 - cst*(PetscReal)(i+j-3);
            if (j>1) PetscCall(MatSetValue(A,ix-1,ix-2,pu,INSERT_VALUES));
            /* west */
            if (i>1) PetscCall(MatSetValue(A,ix-1,ix-jmax-2,pu,INSERT_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(PETSC_SUCCESS);
}
