    #include <slepceps.h>
    #include "restart.h"


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

        Vec            xr,xi;    //need them for eigenvectors printing


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
        PetscCall(EPSSetTolerances(eps, 1e-8, 4)); 
        PetscInt nv, cv,mpd;
        PetscCall(EPSGetDimensions(eps, &nv , &cv, &mpd));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, " nv = %d, cv = %d\n", nv, cv));

        /* SOLVE */
        PetscCall(EPSSolve(eps));


        PetscCall(EPSGetConverged(eps, &nconv));
        PetscBool converged = (nconv > 0);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged? %s\n", converged ? "Yes" : "No"));

        PetscCall(VecDuplicate(v0, &xr));
        PetscCall(VecDuplicate(v0, &xi));

        /* Printing eigenvectors*/
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Printing eigenvectors of eps:\n"));
            for (PetscInt i=0;i<nconv;i++) {
                PetscCall(EPSGetEigenvector(eps,i,xr,xi));
                //PetscCall(VecView(xr,PETSC_VIEWER_STDOUT_WORLD));
            }

        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
        
        PetscReal kr, ki;
        for (PetscInt i=0;i<nconv;i++) {
            PetscCall(EPSGetEigenvalue(eps,i,&kr,&ki));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "eigenvalues eps, Re: %f, Im: %f\n", kr, ki));
        }

        /* Retrieve the BV object from eps.
        This call initializes bv so that it can be used by BVGetSizes, etc. */
        PetscCall(EPSGetBV(eps, &bv));
        PetscInt low, high;
        PetscCall(BVGetSizes(bv, &low , &high, &k));

        //PetscInt low, high;
        //PetscCall(BVGetActiveColumns(bv, &low, &high));
        //k = m;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "BV active columns: low = %d, high = %d\n", k, k));
        k = nv;
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
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Printing all vectors in matrix before saving them V:\n"));
        /*for (PetscInt i = 0; i < k; i++) {
            PetscCall(VecView(V[i], PETSC_VIEWER_STDOUT_WORLD));
        }*/
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
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
        /*PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Printing V_restart after having loaded them:\n"));
        for (PetscInt i = 0; i < k_restart; i++) {
            PetscCall(VecView(V_restart[i], PETSC_VIEWER_STDOUT_WORLD));
        }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));*/



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


        /* Second solving*/
        PetscCall(EPSSolve(eps2));


        /* Check Convergence */
        PetscCall(EPSGetConverged(eps2, &nconv));
        converged = (nconv > 0);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged? %s\n", converged ? "Yes" : "No"));
        
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Printing eigenvectors of eps2:\n"));
            for (PetscInt i=0;i<nconv;i++) {
                PetscCall(EPSGetEigenvector(eps2,i,xr,xi));
                //PetscCall(VecView(xr,PETSC_VIEWER_STDOUT_WORLD));
            }

        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
        /* Formal end*/

        //PetscReal kr, ki;
        for (PetscInt i=0;i<nconv;i++) {
            PetscCall(EPSGetEigenvalue(eps2,i,&kr,&ki));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "eigenvalues, Re: %f, Im: %f\n", kr, ki));
        }

        if  (V_restart) {

            for (PetscInt i = 0; i < k_restart; ++i) {
                if (V_restart[i]){
                    PetscCall(VecDestroy(&V_restart[i]));
                }   
            }
    
            PetscFree(V_restart);
        }

        PetscCall(EPSDestroy(&eps));
        //PetscCall(EPSDestroy(&eps2));
        //PetscCall(MatDestroy(&A));
        //PetscCall(VecDestroy(&v0));
        //PetscCall(BVDestroy(&bv));
        //PetscCall(SlepcFinalize());

        return 0;

    }
