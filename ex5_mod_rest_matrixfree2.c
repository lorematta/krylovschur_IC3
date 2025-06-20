    #include <slepceps.h>
    #include "restart.h"

/* Variant of the standard restart, where EPSSetInitialSpace is not call with the entire restart basis, but whit just the first vector
   It was discovered that actually the function doesn't store the whole basis; which linear combination of the basis'vector should we use?*/

    int main(int argc, char **argv) {
        Vec            v0, v02;
        Mat            A,B;
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
        

        /* Create Matrix-free model */
        PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
        PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
        PetscCall(MatSetFromOptions(A));
        PetscCall(MatMarkovModel(m, A));
        PetscCall(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, &A, &B));
        PetscCall(MatShellSetOperation(B, MATOP_MULT, (void (*)(void))MatMult_MarkovModel));

        /* Create Eigensolver */
        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
        PetscCall(EPSSetOperators(eps, B, NULL));
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
        PetscBool converged = (nconv >= nv);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged? %s\n", converged ? "Yes" : "No"));


        /* Retrieve the BV object from eps.
        This call initializes bv so that it can be used by BVGetSizes, etc. */
        PetscCall(EPSGetBV(eps, &bv));
        PetscInt low, high;
        PetscCall(BVGetSizes(bv, &low , &high, &k));

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
  

        /* Creating eps2*/
        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps2));
        PetscCall(EPSSetOperators(eps2, B, NULL));
        PetscCall(EPSSetProblemType(eps2, EPS_NHEP));
        PetscCall(EPSSetType(eps2, EPSKRYLOVSCHUR));
        PetscCall(EPSSetFromOptions(eps2));



        if (!V_restart || k_restart <=0 ) {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: V_restart is NULL\n"));
            return -1;
        }

        /* Restart is given by considering just the first vector, limit of EPSSetInitialSpace. Is it possible to create a linear combination that optimize that single vector?
           What happen when the krylov subspace is very very wide? is one vector sufficient to carry out a good restart?
            */
        PetscCall(VecDuplicate(V_restart[0], &v02));

        for (PetscInt i =0; i<k; i++){
           // combinazione lineare con tutti coefficienti unitari 
            VecAXPY(v02, 1, V_restart[i]);  

        }
        
        //PetscCall(VecCopy(V_restart[0], v02));
        PetscCall(EPSSetInitialSpace(eps2, 1, &v02));


        /* Second solving*/
        PetscCall(EPSSolve(eps2));


        /* Check Convergence */
        PetscCall(EPSGetConverged(eps2, &nconv));
        converged = (nconv > 0);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged? %s\n", converged ? "Yes" : "No"));
        


        PetscReal kr, ki;
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


        PetscCall(SlepcFinalize());

        return 0;
    }
