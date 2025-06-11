    #include <slepceps.h>
    #include "restart.h"

    /* User-defined routines */
    PetscErrorCode MatMarkovModel(PetscInt m, Mat A);
    PetscErrorCode SaveRestartData(Vec *V, PetscInt k);
    PetscErrorCode LoadRestartData(Vec **V_restart, PetscInt *k_restart, Vec v0);

    int main(int argc, char **argv) {
        Vec            v0;
        Mat            A;
        EPS            eps, eps2;
        BV             bv;
        EPSType        type;
        EPSStop        stop;
        PetscInt       nconv = 0, k = 0; 
        PetscInt       N, m = 15, nev, k_restart = 0;
        Vec           *V_restart = NULL, *V = NULL;
        PetscMPIInt    rank;




        /* Begin*/

        PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));

        /* Problem Setup */
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
        N = m * (m + 1) / 2;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nMarkov Model, N=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n\n", N, m));

        /* Create Matrix */
        PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
        PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
        PetscCall(MatSetFromOptions(A));
        PetscCall(MatMarkovModel(m, A));

        /* Create Eigensolver */
        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
        PetscCall(EPSSetOperators(eps, A, NULL));
        PetscCall(EPSSetProblemType(eps, EPS_NHEP));
        PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR)); // Ensure Krylov-Schur is used
        PetscCall(EPSSetFromOptions(eps));
       
        /* Set Initial Vector */
        PetscCall(MatCreateVecs(A, &v0, NULL));
        PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

        if (!rank) {
            PetscCall(VecSetValue(v0, 0, 1.0, INSERT_VALUES));
            PetscCall(VecSetValue(v0, 1, 1.0, INSERT_VALUES));
            PetscCall(VecSetValue(v0, 2, 1.0, INSERT_VALUES));
        }

        PetscCall(VecAssemblyBegin(v0));
        PetscCall(VecAssemblyEnd(v0));
        PetscCall(EPSSetInitialSpace(eps, 1, &v0));

        /* Force a single iteration */
        //PetscCall(EPSSetTolerances(eps, 1e-2, 1)); 
        PetscInt nv, cv,mpd;
        PetscCall(EPSGetDimensions(eps, &nv , &cv, &mpd));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, " nv = %d, cv = %d\n", nv, cv));

        /* SOLVE */
        PetscCall(EPSSolve(eps));


        /* Retrieve the BV object from eps.
        This call initializes bv so that it can be used by BVGetSizes, etc. */
        PetscCall(EPSGetBV(eps, &bv));
        PetscInt low, high;
        PetscCall(BVGetSizes(bv, &low , &high, &k));

        //PetscInt low, high;
        //PetscCall(BVGetActiveColumns(bv, &low, &high));
        //k = m;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "BV active columns: low = %d, high = %d\n", k, k));
        k = 2;
        PetscCall(PetscMalloc1(k, &V));

        for (PetscInt i = 0; i < k; i++) {
            Vec vi;
            /* Retrieve the i-th column from the BV */
            PetscCall(BVGetColumn(bv,  i, &vi));
            
            /* Duplicate the column so that it is independent of the BV object */
            PetscCall(VecDuplicate(vi, &V[i]));
            PetscCall(VecCopy(vi, V[i]));
      
            /* Restore the column after you're done making your copy */
            PetscCall(BVRestoreColumn(bv,  i, &vi));
        }

        /* Print*/
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Printing all vectors in matrix V:\n"));
        for (PetscInt i = 0; i < k; i++) {
            PetscCall(VecView(V[i], PETSC_VIEWER_STDOUT_WORLD));
        }
        /* Save Restart Data */
        PetscCall(SaveRestartData(V, k));


        /* Cleanup */
        for (PetscInt i = 0; i < k; ++i) {
            PetscCall(VecDestroy(&V[i]));
        }
        PetscFree(V);

        /* Load Restart Data */
        PetscCall(LoadRestartData(&V_restart, &k_restart, v0));
        /* Print*/
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Printing V_restart:\n"));
        for (PetscInt i = 0; i < k_restart; i++) {
            PetscCall(VecView(V_restart[i], PETSC_VIEWER_STDOUT_WORLD));
        }


        /* Creating eps2*/
        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps2));
        PetscCall(EPSSetOperators(eps2, A, NULL));
        PetscCall(EPSSetProblemType(eps2, EPS_NHEP));
        PetscCall(EPSSetType(eps2, EPSKRYLOVSCHUR));
        PetscCall(EPSSetFromOptions(eps2));



        if (!V_restart || k_restart <=0 ) {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: V_restart is NULL\n"));
            return -1;
        }

        PetscCall(EPSSetInitialSpace(eps2, k_restart, V_restart));
        //PetscCall(EPSSetInitialSpace(eps2, 1, &v0));

        /* Second solving*/
        PetscCall(EPSSolve(eps2));


        /* Check Convergence */
        PetscCall(EPSGetConverged(eps2, &nconv));
        PetscBool converged = (nconv > 0);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged? %s\n", converged ? "Yes" : "No"));

        /* Formal end*/

        if  (V_restart) {

            for (PetscInt i = 0; i < k_restart; ++i) {
                if (V_restart[i]){
                    PetscCall(VecDestroy(&V_restart[i]));
                }   
            }
    
            PetscFree(V_restart);
        }

        //PetscCall(EPSDestroy(&eps));
        //PetscCall(EPSDestroy(&eps2));
        //PetscCall(MatDestroy(&A));
        //PetscCall(VecDestroy(&v0));
        //PetscCall(BVDestroy(&bv));
        //PetscCall(SlepcFinalize());

        return 0;

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
