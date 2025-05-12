# krylovschur_IC3

I'm trying to implement the Krylov-Schur method to investigate eigenvalues of instability fluidmechanics problem. In particular, i'm interested to investigate the problem emerged in usage of slepc libraries in a supercomputer, left to work for 24 hours straight.
After every cycle, at the moment, the algorithm is not capable of find any eigenvalue and all the proccess is useful.
The aims are the following: either modify the library in order to save the results of the first cylce so that the second one can restart not from 0, either rewrite the algorithm so that we can work directly on the main code 
