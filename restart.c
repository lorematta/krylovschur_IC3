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
    void              *ctx;
    PetscInt          nx, i, j, jmax, ix = 0;
    const PetscScalar *px;
    PetscScalar       *py;
    PetscReal         cst;

    PetscFunctionBeginUser;

    /* Retrieve the size context from MatShell */
    PetscCall(MatShellGetContext(A, &ctx));
    if (!ctx) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: MatShell context is NULL!\n"));
        PetscFunctionReturn(PETSC_ERR_ARG_WRONG);
    }
    nx = *((PetscInt*)ctx);
    if (nx <= 0) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: Invalid nx value (%d)\n", nx));
        PetscFunctionReturn(PETSC_ERR_ARG_WRONG);
    }
    cst = 0.5 / (PetscReal)(nx - 1);

    /* Get vector arrays */
    PetscCall(VecGetArrayRead(x, &px));
    PetscCall(VecGetArray(y, &py));

    /* Initialize y vector */
    PetscCall(VecSet(y, 0.0));

    /* Apply matrix-vector multiplication */
    for (i = 1; i <= nx; i++) {
        jmax = nx - i + 1;
        for (j = 1; j <= jmax; j++) {
            ix++;

            if (ix >= nx) continue; // Prevent out-of-bounds errors

            /* Compute transition probabilities */
            PetscReal pd = cst * (PetscReal)(i + j - 1);
            PetscReal pu = 0.5 - cst * (PetscReal)(i + j - 3);

            /* North */
            if (i == 1 && ix < nx) py[ix - 1] += 2 * pd * px[ix];
            else if (ix < nx) py[ix - 1] += pd * px[ix];

            /* East */
            if (j == 1 && (ix + jmax - 1) < nx) py[ix - 1] += 2 * pd * px[ix + jmax - 1];
            else if ((ix + jmax - 1) < nx) py[ix - 1] += pd * px[ix + jmax - 1];

            /* South */
            if (j > 1 && ix - 2 >= 0) py[ix - 1] += pu * px[ix - 2];

            /* West */
            if (i > 1 && ix - jmax - 2 >= 0) py[ix - 1] += pu * px[ix - jmax - 2];
        }
    }

    /* Restore vector arrays */
    PetscCall(VecRestoreArrayRead(x, &px));
    PetscCall(VecRestoreArray(y, &py));

    PetscFunctionReturn(PETSC_SUCCESS);
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
