    #include <slepceps.h>
    #include "restart.h"

    /* User-defined routines */
    PetscErrorCode MatMarkovModel(PetscInt m, Mat A);
    PetscErrorCode SaveRestartData(Vec *V, PetscInt k);
    PetscErrorCode LoadRestartData(Vec **V_restart, PetscInt *k_restart, Vec v0);

    int main(int argc, char **argv) {
        Vec            v0;
        Mat            A;
        EPS            eps;
        EPS            eps2;
        EPSType        type;
        EPSStop        stop;
        PetscInt       nconv = 0; 
        PetscReal      thres;
        PetscInt       N, m = 15, nev, k_restart;
        Vec           *V_restart;
        PetscMPIInt    rank;
        PetscBool      terse;

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
        PetscCall(EPSSetTolerances(eps, PETSC_DEFAULT, 1)); 



        PetscCall(EPSSolve(eps));

        /* Extract Basis for Restart */
        PetscInt k = m;
        Vec *V;
        PetscCall(PetscMalloc1(k, &V));

        for (PetscInt i = 0; i < k; i++) {
            PetscCall(VecDuplicate(v0, &V[i]));
            PetscCall(EPSGetEigenvector(eps, i, V[i], NULL));
        }

        /* Save Restart Data */
        PetscCall(SaveRestartData(V, k));

        /* Cleanup */
        for (PetscInt i = 0; i < k; ++i) PetscCall(VecDestroy(&V[i]));
        PetscFree(V);

        /* Load Restart Data */
        PetscCall(LoadRestartData(&V_restart, &k_restart, v0));

        /* Creating eps2*/
        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps2));
        PetscCall(EPSSetOperators(eps2, A, NULL));
        PetscCall(EPSSetProblemType(eps2, EPS_NHEP));
        PetscCall(EPSSetType(eps2, EPSKRYLOVSCHUR));
        PetscCall(EPSSetFromOptions(eps2));

        if (!V_restart || !k_restart <=0 ) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: V_restart is NULL\n"));
        return -1;
        }

        PetscCall(EPSSetInitialSpace(eps2, k_restart, V_restart));

        /* Second solving*/
        PetscCall(EPSSolve(eps2));


        /* Check Convergence */
        PetscCall(EPSGetConverged(eps2, &nconv));
        PetscBool converged = (nconv > 0);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iteration %" PetscInt_FMT ": Converged? %s\n", converged ? "Yes" : "No"));

        /* Formal end*/

        if  (V_restart) {

            for (PetscInt i = 0; i < k_restart; ++i) {
                if (V_restart[i])   PetscCall(VecDestroy(&V_restart[i]));
            }
    
            PetscFree(V_restart);
        }

        PetscCall(EPSDestroy(&eps));
        PetscCall(EPSDestroy(&eps2));
        PetscCall(MatDestroy(&A));
        PetscCall(VecDestroy(&v0));

        return 0;

    }
