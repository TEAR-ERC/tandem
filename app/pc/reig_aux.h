#ifndef REIG_AUX_20210209_H
#define REIG_AUX_20210209_H

#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

PetscErrorCode MatOrthogonalizeModifiedGramSchmidt(PetscInt n, Vec X[]);
PetscErrorCode MatOrthogonalizeMatDense(Mat Q);
PetscErrorCode MatDenseCopyFast(Mat A, Mat B);
PetscErrorCode MatSetRandomConsistent(Mat Q);
PetscErrorCode MatMultRedundant_MatDenseVecSeq(Mat A, Vec x, Vec y);
PetscErrorCode MatDenseSolveKSP(KSP ksp, Mat X, MatReuse scall, Mat* _Y);
PetscErrorCode RandEigsMax_SEQ_2(Mat A, PetscInt k, PetscInt o, PetscInt power_its,
                                 PetscRandom prand, Vec* _eigs, Mat* _V);
PetscErrorCode RandEigsMax_MPI_2(Mat A, PetscInt k, PetscInt o, PetscInt power_its,
                                 PetscRandom prand, Vec* _eigs, Mat* _V);
PetscErrorCode RandEigsMin_SEQ_2(KSP ksp, PetscInt k, PetscInt o, PetscInt power_its,
                                 PetscRandom prand, Vec* _eigs, Mat* _V);
PetscErrorCode RandEigsMin_MPI_2(KSP ksp, PetscInt k, PetscInt o, PetscInt power_its,
                                 PetscRandom prand, Vec* _eigs, Mat* _V);

#endif // REIG_AUX_20210209_H
