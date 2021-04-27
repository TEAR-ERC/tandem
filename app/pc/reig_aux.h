#ifndef REIG_AUX_20210209_H
#define REIG_AUX_20210209_H

#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

PetscErrorCode VecOrthonormalize_ModifiedGramSchmidt(PetscInt, const Vec*);
PetscErrorCode VecOrthonormalize(PetscInt, const Vec*);
PetscErrorCode MatDenseOrthonormalize(Mat);

/* Default PETSc MatCopy() was slow in debug build - this is faster */
PetscErrorCode MatDenseCopyFast(Mat, Mat);

/* Similar functionality is provided in petsc 3.14 */
PetscErrorCode _MatDenseCreateEmptyColumnVec(Mat, Vec*);
PetscErrorCode _MatDenseSetColumnVec(Mat, PetscInt, Vec);
PetscErrorCode _MatDenseResetColumnVec(Mat, PetscInt, Vec);

/* Helpers to define random matrices in a manner independent of # of MPI ranks */
PetscErrorCode MatSetRandomConsistent(Mat);
PetscErrorCode MatDenseSetRandomConsistent(Mat);

/* Special MatMult method for an input redundant SEQ vector */
PetscErrorCode MatMultRedundant_MatDenseVecSeq(Mat, Vec, Vec);

/* MatMatMult variants for MATDENSE */
PetscErrorCode MatMatMult_MatDense(Mat, Mat, MatReuse, Mat*);
PetscErrorCode KSPMatMult_MatDense(KSP, Mat, MatReuse, Mat*);
PetscErrorCode MatMatMult_MatDense_InPlace(Mat, Mat);
PetscErrorCode KSPMatMult_MatDense_InPlace(KSP, Mat, PetscBool);

/* MatPtAP variants for MATDENSE */
PetscErrorCode MatPtAP_MatDense(Mat, Mat, MatReuse, Mat*);
PetscErrorCode MatPtInvAP_MatDense(KSP, Mat, MatReuse, Mat*);

PetscErrorCode MatDenseSymmetrizeSEQ(Mat);

/*
  Randomized eigenvalue/eigenvector methods.

  RandEigsMax() Computes an approximate truncated rank-k eigendecomposition associated with the k
  largest eigenvalues. RandEigsMin() Computes an approximate truncated rank-k eigendecomposition
  associated with the k smallest eigenvalues.

  Input
    A/ksp - The operator, dimensions M x M
    k     - requested rank of the eigendecomposition.
    o     - number of columns used for over-sampling (values larger than 0 will improve accuracy).
    pits  - number of power-iterations performed (values larger than 0 will improve accuracy).
    prand - PetscRandom object, can be NULL.

  Output
    _eigs - Vector (VECSEQ) of eigenvalues of dimension k.
            The resulting vector is redundant, i.e. duplicated over all MPI ranks, and will be of
  type VECSEQ. Must be non-NULL. _V   - Matrix (MATDENSE) of eigenvectors of dimension M x k.
            Optional argument, can be ignored if you pass NULL.

  Notes
    - Both RandEigsMax() and RandEigsMin() returns eigenvalues (and associated eigenvectors) sorted
  according to their magnitude, from smallest to largest.
    - RandEigsMin() uses a shift-and-invert strategy to define the k smallest eigenvalues.
    - Both implementations require storing a single auxillary (dense) matrix of dimensions M x
  (k+o).
    - RandEigsMin() enforces symmetry of the reduced system.
*/
PetscErrorCode RandEigsMax(Mat A, PetscInt k, PetscInt o, PetscInt pits, PetscRandom prand,
                           Vec* _eigs, Mat* _V);
PetscErrorCode RandEigsMin(KSP ksp, PetscInt k, PetscInt o, PetscInt pits, PetscRandom prand,
                           Vec* _eigs, Mat* _V);

#endif // REIG_AUX_20210209_H
