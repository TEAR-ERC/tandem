#include "reig_aux.h"
#include "common/lapack.h"

PetscErrorCode VecOrthonormalize_ModifiedGramSchmidt(PetscInt n, const Vec X[]) {
    PetscErrorCode ierr;
    Vec q, v;
    PetscInt j, k;
    PetscReal nrm, dot;

    PetscFunctionBegin;
    for (j = 0; j < n; j++) {
        q = X[j];
        ierr = VecNorm(q, NORM_2, &nrm);
        CHKERRQ(ierr);
        ierr = VecScale(q, 1.0 / nrm);
        CHKERRQ(ierr);
        for (k = j + 1; k < n; k++) {
            v = X[k];
            ierr = VecDot(q, v, &dot);
            CHKERRQ(ierr);
            ierr = VecAXPY(v, -dot, q);
            CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode VecOrthonormalize(PetscInt n, const Vec X[]) {
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = VecOrthonormalize_ModifiedGramSchmidt(n, X);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode MatDenseOrthonormalize(Mat Q) {
    PetscErrorCode ierr;
    PetscInt j, M, m, N, bs;
    Vec* q;
    PetscScalar* array;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)Q, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)Q, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)Q), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
#endif

    /* Create array of vectors */
    ierr = MatGetSize(Q, &M, &N);
    CHKERRQ(ierr);
    ierr = MatGetLocalSize(Q, &m, NULL);
    CHKERRQ(ierr);
    ierr = MatGetBlockSize(Q, &bs);
    CHKERRQ(ierr);

    ierr = MatDenseGetArray(Q, &array);
    CHKERRQ(ierr);

    ierr = PetscCalloc1(N, &q);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        PetscScalar* array_j = array + j * m;
        ierr =
            VecCreateMPIWithArray(PETSC_COMM_WORLD, bs, m, M, (const PetscScalar*)array_j, &q[j]);
        CHKERRQ(ierr);
    }

    ierr = MatDenseRestoreArray(Q, &array);
    CHKERRQ(ierr);

    ierr = VecOrthonormalize(N, (const Vec*)q);
    CHKERRQ(ierr);

    for (j = 0; j < N; j++) {
        ierr = VecDestroy(&q[j]);
        CHKERRQ(ierr);
    }
    ierr = PetscFree(q);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode MatDenseCopyFast(Mat A, Mat B) {
    PetscErrorCode ierr;
    PetscInt m, n;
    Mat lA;
    PetscScalar *_a, *_b;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
#endif

    ierr = MatDenseGetLocalMatrix(A, &lA);
    CHKERRQ(ierr);
    ierr = MatGetSize(lA, &m, &n);
    CHKERRQ(ierr);
    ierr = MatDenseGetArray(A, &_a);
    CHKERRQ(ierr);
    ierr = MatDenseGetArray(B, &_b);
    CHKERRQ(ierr);
    ierr = PetscMemcpy(_b, _a, sizeof(PetscScalar) * m * n);
    CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(B, &_a);
    CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(A, &_b);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode _MatDenseCreateEmptyColumnVec(Mat A, Vec* v) {
    PetscErrorCode ierr;
    PetscInt M, N, m, n, bs;
    PetscMPIInt commsize;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
#endif

    ierr = MatGetSize(A, &M, &N);
    CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &m, &n);
    CHKERRQ(ierr);
    ierr = MatGetBlockSize(A, &bs);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A), &commsize);
    CHKERRQ(ierr);
    if (commsize == 1) {
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, bs, M, NULL, v);
        CHKERRQ(ierr);
    } else {
        ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A), bs, m, M, NULL, v);
        CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode _MatDenseSetColumnVec(Mat A, PetscInt j, Vec v) {
    PetscErrorCode ierr;
    PetscInt lda;
    PetscScalar* array;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
#endif

    ierr = MatDenseGetLDA(A, &lda);
    CHKERRQ(ierr);
    ierr = MatDenseGetArray(A, &array);
    CHKERRQ(ierr);
    ierr = VecPlaceArray(v, array + (size_t)j * (size_t)lda);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode _MatDenseResetColumnVec(Mat A, PetscInt j, Vec v) {
    PetscErrorCode ierr;
    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
#endif
    ierr = VecResetArray(v);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*
  Creates a random matrix which is indepednent of the partition size (# of MPI ranks)
  Q can be of any MatType provided it supports MatSetValue()
*/
PetscErrorCode MatSetRandomConsistent(Mat Q) {
    PetscErrorCode ierr;
    Vec q;
    PetscInt M, N, j, i, start, end;
    PetscRandom randObj;
    const PetscScalar* _q;

    PetscFunctionBegin;
    ierr = MatZeroEntries(Q);
    CHKERRQ(ierr);
    ierr = MatGetSize(Q, &M, &N);
    CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &q);
    CHKERRQ(ierr);
    ierr = VecSetSizes(q, PETSC_DECIDE, M);
    CHKERRQ(ierr);
    ierr = VecSetType(q, VECSEQ);
    CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(Q, &start, &end);
    CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &randObj);
    CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randObj);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        ierr = VecSetRandom(q, randObj);
        CHKERRQ(ierr);
        ierr = VecGetArrayRead(q, &_q);
        CHKERRQ(ierr);
        for (i = start; i < end; i++) {
            ierr = MatSetValue(Q, i, j, _q[i], INSERT_VALUES);
            CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(q, &_q);
        CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    ierr = PetscRandomDestroy(&randObj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&q);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*
 Creates a random matrix which is indepednent of the partition size (# of MPI ranks)
 Q must be of MatType MATDENSE
 */
PetscErrorCode MatDenseSetRandomConsistent(Mat Q) {
    PetscErrorCode ierr;
    Vec q;
    PetscInt m, M, N, j, i, start, end;
    PetscRandom randObj;
    const PetscScalar* _q;
    PetscScalar* _Q;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)Q, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)Q, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)Q), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
#endif

    ierr = MatGetSize(Q, &M, &N);
    CHKERRQ(ierr);
    ierr = MatGetLocalSize(Q, &m, NULL);
    CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_SELF, &q);
    CHKERRQ(ierr);
    ierr = VecSetSizes(q, PETSC_DECIDE, M);
    CHKERRQ(ierr);
    ierr = VecSetType(q, VECSEQ);
    CHKERRQ(ierr);

    ierr = PetscRandomCreate(PETSC_COMM_SELF, &randObj);
    CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randObj);
    CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(Q, &start, &end);
    CHKERRQ(ierr);
    ierr = MatDenseGetArray(Q, &_Q);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        ierr = VecSetRandom(q, randObj);
        CHKERRQ(ierr);
        ierr = VecGetArrayRead(q, &_q);
        CHKERRQ(ierr);
        for (i = start; i < end; i++) {
            _Q[(i - start) + j * m] = _q[i];
        }
        ierr = VecRestoreArrayRead(q, &_q);
        CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(Q, &_Q);
    CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    ierr = PetscRandomDestroy(&randObj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&q);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*
  Computes
    y = A x
  where
    A is MATDENSE (SEQ or MPI)
    x is VECSEQ (redundant across all ranks)
    y is VECMPI
*/
PetscErrorCode MatMultRedundant_MatDenseVecSeq(Mat A, Vec x, Vec y) {
    PetscErrorCode ierr;
    Mat lA;
    Vec ly;
    PetscInt m, bs;
    PetscScalar* _y;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        PetscBool isseq = PETSC_FALSE;
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
        ierr = PetscObjectTypeCompare((PetscObject)x, VECSEQ, &isseq);
        CHKERRQ(ierr);
        if (!isseq)
            SETERRQ(PetscObjectComm((PetscObject)x), PETSC_ERR_SUP,
                    "Only valid for VecType VECSEQ");
    }
#endif

    ierr = VecGetLocalSize(y, &m);
    CHKERRQ(ierr);
    ierr = VecGetBlockSize(x, &bs);
    CHKERRQ(ierr);
    ierr = VecGetArray(y, &_y);
    CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, bs, m, _y, &ly);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &_y);
    CHKERRQ(ierr);
    ierr = MatDenseGetLocalMatrix(A, &lA);
    CHKERRQ(ierr);
    ierr = MatMult(lA, x, ly);
    CHKERRQ(ierr);
    ierr = VecDestroy(&ly);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*
  Y = A X

  A and X must of type MATDENSE
  Y will / must be of type MATDENSE
*/
PetscErrorCode MatMatMult_MatDense(Mat A, Mat X, MatReuse scall, Mat* _Y) {
    PetscErrorCode ierr;
    PetscInt j, N;
    Vec Xj, Yj;
    Mat Y;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)X, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)X, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)X), PETSC_ERR_SUP,
                    "Only valid for X[MatType] MATSEQDENSE or MATMPIDENSE");
    }
#endif

    if (scall == MAT_INITIAL_MATRIX) {
        PetscInt m, n, M, N;

        ierr = MatGetSize(X, &M, &N);
        CHKERRQ(ierr);
        ierr = MatGetLocalSize(X, &m, &n);
        CHKERRQ(ierr);
        ierr = MatCreate(PetscObjectComm((PetscObject)X), &Y);
        CHKERRQ(ierr);
        ierr = MatSetSizes(Y, m, n, M, N);
        CHKERRQ(ierr);
        ierr = MatSetType(Y, MATDENSE);
        CHKERRQ(ierr);
        ierr = MatSetUp(Y);
        CHKERRQ(ierr);
        *_Y = Y;
    } else if (scall == MAT_REUSE_MATRIX) {
        Y = *_Y;
#if defined(PETSC_USE_DEBUG)
        {
            PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
            ierr = PetscObjectTypeCompare((PetscObject)Y, MATSEQDENSE, &isdense[0]);
            CHKERRQ(ierr);
            ierr = PetscObjectTypeCompare((PetscObject)Y, MATMPIDENSE, &isdense[1]);
            CHKERRQ(ierr);
            if (!isdense[0] && !isdense[1])
                SETERRQ(PetscObjectComm((PetscObject)Y), PETSC_ERR_SUP,
                        "Only valid for Y[MatType] MATSEQDENSE or MATMPIDENSE");
        }
#endif

    } else
        SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
                "Only support for MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX");

    ierr = MatZeroEntries(Y);
    CHKERRQ(ierr);

    ierr = _MatDenseCreateEmptyColumnVec(X, &Xj);
    CHKERRQ(ierr);
    ierr = _MatDenseCreateEmptyColumnVec(Y, &Yj);
    CHKERRQ(ierr);

    ierr = MatGetSize(X, NULL, &N);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        ierr = _MatDenseSetColumnVec(X, j, Xj);
        CHKERRQ(ierr);
        ierr = _MatDenseSetColumnVec(Y, j, Yj);
        CHKERRQ(ierr);
        ierr = MatMult(A, Xj, Yj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(Y, j, Yj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(X, j, Xj);
        CHKERRQ(ierr);
    }

    ierr = VecDestroy(&Xj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Yj);
    CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/*
  Y = inv(A) X
  X must of type MATDENSE
  Y will / must be of type MATDENSE

  Solve
    A Y[i] = X[i]
  for all columns i in X
*/
PetscErrorCode KSPMatMult_MatDense(KSP ksp, Mat X, MatReuse scall, Mat* _Y) {
    PetscErrorCode ierr;
    PetscInt j, N;
    Vec Xj, Yj;
    Mat Y;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)X, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)X, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)X), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
#endif

    if (scall == MAT_INITIAL_MATRIX) {
        PetscInt m, n, M, N;

        ierr = MatGetSize(X, &M, &N);
        CHKERRQ(ierr);
        ierr = MatGetLocalSize(X, &m, &n);
        CHKERRQ(ierr);
        ierr = MatCreate(PetscObjectComm((PetscObject)X), &Y);
        CHKERRQ(ierr);
        ierr = MatSetSizes(Y, m, n, M, N);
        CHKERRQ(ierr);
        ierr = MatSetType(Y, MATDENSE);
        CHKERRQ(ierr);
        ierr = MatSetUp(Y);
        CHKERRQ(ierr);
        *_Y = Y;
    } else if (scall == MAT_REUSE_MATRIX) {
        Y = *_Y;
#if defined(PETSC_USE_DEBUG)
        {
            PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
            ierr = PetscObjectTypeCompare((PetscObject)Y, MATSEQDENSE, &isdense[0]);
            CHKERRQ(ierr);
            ierr = PetscObjectTypeCompare((PetscObject)Y, MATMPIDENSE, &isdense[1]);
            CHKERRQ(ierr);
            if (!isdense[0] && !isdense[1])
                SETERRQ(PetscObjectComm((PetscObject)Y), PETSC_ERR_SUP,
                        "Only valid for Y[MatType] MATSEQDENSE or MATMPIDENSE");
        }
#endif
    } else
        SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP,
                "Only support for MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX");

#if 0
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
#endif

    ierr = MatZeroEntries(Y);
    CHKERRQ(ierr);

    ierr = _MatDenseCreateEmptyColumnVec(X, &Xj);
    CHKERRQ(ierr);
    ierr = _MatDenseCreateEmptyColumnVec(Y, &Yj);
    CHKERRQ(ierr);

    ierr = MatGetSize(X, NULL, &N);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        ierr = _MatDenseSetColumnVec(X, j, Xj);
        CHKERRQ(ierr);
        ierr = _MatDenseSetColumnVec(Y, j, Yj);
        CHKERRQ(ierr);
        /* using last column as guess - why does this work at ? */
#if 0
    if (j > 0) {
      Vec Yjlast;
      ierr = _MatDenseCreateEmptyColumnVec(Y,&Yjlast);CHKERRQ(ierr);
      ierr = _MatDenseSetColumnVec(Y,j-1,Yjlast);CHKERRQ(ierr);
      ierr = VecCopy(Yjlast,Yj);CHKERRQ(ierr);
      ierr = VecDestroy(&Yjlast);CHKERRQ(ierr);
    }
#endif
        // ierr = MatDenseGetColumnVec(X,j,&Xj);CHKERRQ(ierr);
        // ierr = MatDenseGetColumnVec(Y,j,&Yj);CHKERRQ(ierr);
        ierr = KSPSolve(ksp, Xj, Yj);
        CHKERRQ(ierr);
        // ierr = MatDenseRestoreColumnVec(Y,j,&Yj);CHKERRQ(ierr);
        // ierr = MatDenseRestoreColumnVec(X,j,&Xj);CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(Y, j, Yj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(X, j, Xj);
        CHKERRQ(ierr);
    }

    ierr = VecDestroy(&Xj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Yj);
    CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/*
  X = A X
  X must of type MATDENSE
*/
PetscErrorCode MatMatMult_MatDense_InPlace(Mat A, Mat X) {
    PetscErrorCode ierr;
    PetscInt j, N;
    Vec Xj, Yj;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)X, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)X, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)X), PETSC_ERR_SUP,
                    "Only valid for X[MatType] MATSEQDENSE or MATMPIDENSE");
    }
#endif

    ierr = _MatDenseCreateEmptyColumnVec(X, &Xj);
    CHKERRQ(ierr);
    ierr = MatCreateVecs(X, NULL, &Yj);
    CHKERRQ(ierr);

    ierr = MatGetSize(X, NULL, &N);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        ierr = _MatDenseSetColumnVec(X, j, Xj);
        CHKERRQ(ierr);
        ierr = MatMult(A, Xj, Yj);
        CHKERRQ(ierr);
        ierr = VecCopy(Yj, Xj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(X, j, Xj);
        CHKERRQ(ierr);
    }

    ierr = VecDestroy(&Xj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Yj);
    CHKERRQ(ierr);

    ierr = MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/*
  X = inv(A) X
  X must of type MATDENSE
*/
PetscErrorCode KSPMatMult_MatDense_InPlace(KSP ksp, Mat X, PetscBool use_nonzero_guess) {
    PetscErrorCode ierr;
    PetscInt j, N;
    Vec Xj, Yj;
    PetscBool flg;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)X, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)X, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)X), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
#endif

    ierr = KSPGetInitialGuessNonzero(ksp, &flg);
    CHKERRQ(ierr);
    if (use_nonzero_guess) {
        ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
        CHKERRQ(ierr);
    }

    ierr = _MatDenseCreateEmptyColumnVec(X, &Xj);
    CHKERRQ(ierr);
    ierr = MatCreateVecs(X, NULL, &Yj);
    CHKERRQ(ierr);

    ierr = MatGetSize(X, NULL, &N);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        ierr = _MatDenseSetColumnVec(X, j, Xj);
        CHKERRQ(ierr);
        if (use_nonzero_guess) {
            ierr = VecCopy(Xj, Yj);
            CHKERRQ(ierr);
        }
        ierr = KSPSolve(ksp, Xj, Yj);
        CHKERRQ(ierr);
        ierr = VecCopy(Yj, Xj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(X, j, Xj);
        CHKERRQ(ierr);
    }

    ierr = VecDestroy(&Xj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Yj);
    CHKERRQ(ierr);

    ierr = MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    if (use_nonzero_guess) {
        ierr = KSPSetInitialGuessNonzero(ksp, flg);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

/*
  Y = Pt.A.P
  P must be of type MATDENSE
  Y will/must be of type MATDENSE
*/
PetscErrorCode MatPtAP_MatDense(Mat A, Mat P, MatReuse scall, Mat* _Y) {
    PetscErrorCode ierr;
    PetscInt j, N;
    Vec Pj, APj, Ptj, Yj;
    Mat Y;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)P, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)P, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)P), PETSC_ERR_SUP,
                    "Only valid for P[MatType] MATSEQDENSE or MATMPIDENSE");
    }
#endif

    if (scall == MAT_INITIAL_MATRIX) {
        PetscInt m, n, M, N;

        ierr = MatGetSize(P, &M, &N);
        CHKERRQ(ierr);
        ierr = MatGetLocalSize(P, &m, &n);
        CHKERRQ(ierr);
        ierr = MatCreate(PetscObjectComm((PetscObject)P), &Y);
        CHKERRQ(ierr);
        ierr = MatSetSizes(Y, n, n, N, N);
        CHKERRQ(ierr);
        ierr = MatSetType(Y, MATDENSE);
        CHKERRQ(ierr);
        ierr = MatSetUp(Y);
        CHKERRQ(ierr);
        *_Y = Y;
    } else if (scall == MAT_REUSE_MATRIX) {
        Y = *_Y;
#if defined(PETSC_USE_DEBUG)
        {
            PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
            ierr = PetscObjectTypeCompare((PetscObject)Y, MATSEQDENSE, &isdense[0]);
            CHKERRQ(ierr);
            ierr = PetscObjectTypeCompare((PetscObject)Y, MATMPIDENSE, &isdense[1]);
            CHKERRQ(ierr);
            if (!isdense[0] && !isdense[1])
                SETERRQ(PetscObjectComm((PetscObject)Y), PETSC_ERR_SUP,
                        "Only valid for Y[MatType] MATSEQDENSE or MATMPIDENSE");
        }
#endif

        ierr = MatZeroEntries(Y);
        CHKERRQ(ierr);
    } else
        SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
                "Only support for MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX");

    ierr = _MatDenseCreateEmptyColumnVec(P, &Pj);
    CHKERRQ(ierr);
    ierr = _MatDenseCreateEmptyColumnVec(Y, &Yj);
    CHKERRQ(ierr);
    ierr = MatCreateVecs(P, NULL, &APj);
    CHKERRQ(ierr);
    ierr = MatCreateVecs(P, &Ptj, NULL);
    CHKERRQ(ierr);

    ierr = MatGetSize(P, NULL, &N);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        ierr = _MatDenseSetColumnVec(P, j, Pj);
        CHKERRQ(ierr);
        ierr = MatMult(A, Pj, APj);
        CHKERRQ(ierr);
        ierr = MatMultTranspose(P, APj, Ptj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(P, j, Pj);
        CHKERRQ(ierr);

        ierr = _MatDenseSetColumnVec(Y, j, Yj);
        CHKERRQ(ierr);
        ierr = VecCopy(Ptj, Yj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(Y, j, Yj);
        CHKERRQ(ierr);
    }

    ierr = VecDestroy(&Yj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Pj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&APj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Ptj);
    CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/*
  Y = Pt.inv(A).P
  P must of type MATDENSE
  Y must/will be of type MATDENSE
*/
PetscErrorCode MatPtInvAP_MatDense(KSP ksp, Mat P, MatReuse scall, Mat* _Y) {
    PetscErrorCode ierr;
    PetscInt j, N;
    Vec Pj, APj, Ptj, Yj;
    Mat Y;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
        ierr = PetscObjectTypeCompare((PetscObject)P, MATSEQDENSE, &isdense[0]);
        CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)P, MATMPIDENSE, &isdense[1]);
        CHKERRQ(ierr);
        if (!isdense[0] && !isdense[1])
            SETERRQ(PetscObjectComm((PetscObject)P), PETSC_ERR_SUP,
                    "Only valid for P[MatType] MATSEQDENSE or MATMPIDENSE");
    }
#endif

    if (scall == MAT_INITIAL_MATRIX) {
        PetscInt m, n, M, N;

        ierr = MatGetSize(P, &M, &N);
        CHKERRQ(ierr);
        ierr = MatGetLocalSize(P, &m, &n);
        CHKERRQ(ierr);
        ierr = MatCreate(PetscObjectComm((PetscObject)P), &Y);
        CHKERRQ(ierr);
        ierr = MatSetSizes(Y, n, n, N, N);
        CHKERRQ(ierr);
        ierr = MatSetType(Y, MATDENSE);
        CHKERRQ(ierr);
        ierr = MatSetUp(Y);
        CHKERRQ(ierr);
        *_Y = Y;
    } else if (scall == MAT_REUSE_MATRIX) {
        Y = *_Y;
#if defined(PETSC_USE_DEBUG)
        {
            PetscBool isdense[] = {PETSC_FALSE, PETSC_FALSE};
            ierr = PetscObjectTypeCompare((PetscObject)Y, MATSEQDENSE, &isdense[0]);
            CHKERRQ(ierr);
            ierr = PetscObjectTypeCompare((PetscObject)Y, MATMPIDENSE, &isdense[1]);
            CHKERRQ(ierr);
            if (!isdense[0] && !isdense[1])
                SETERRQ(PetscObjectComm((PetscObject)Y), PETSC_ERR_SUP,
                        "Only valid for Y[MatType] MATSEQDENSE or MATMPIDENSE");
        }
#endif
        ierr = MatZeroEntries(Y);
        CHKERRQ(ierr);
    } else
        SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP,
                "Only support for MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX");

    ierr = _MatDenseCreateEmptyColumnVec(P, &Pj);
    CHKERRQ(ierr);
    ierr = _MatDenseCreateEmptyColumnVec(Y, &Yj);
    CHKERRQ(ierr);
    ierr = MatCreateVecs(P, NULL, &APj);
    CHKERRQ(ierr);
    ierr = MatCreateVecs(P, &Ptj, NULL);
    CHKERRQ(ierr);

    ierr = MatGetSize(P, NULL, &N);
    CHKERRQ(ierr);
    for (j = 0; j < N; j++) {
        ierr = _MatDenseSetColumnVec(P, j, Pj);
        CHKERRQ(ierr);
        ierr = VecZeroEntries(APj);
        CHKERRQ(ierr);
        ierr = KSPSolve(ksp, Pj, APj);
        CHKERRQ(ierr);
        ierr = MatMultTranspose(P, APj, Ptj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(P, j, Pj);
        CHKERRQ(ierr);

        ierr = _MatDenseSetColumnVec(Y, j, Yj);
        CHKERRQ(ierr);
        ierr = VecCopy(Ptj, Yj);
        CHKERRQ(ierr);
        ierr = _MatDenseResetColumnVec(Y, j, Yj);
        CHKERRQ(ierr);
    }

    ierr = VecDestroy(&Yj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Pj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&APj);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Ptj);
    CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/*
  A <- 0.5(A + A^T)
  A must of type MATSEQDENSE
*/
PetscErrorCode MatDenseSymmetrizeSEQ(Mat A) {
    PetscScalar* _A;
    PetscInt m, i, j;
    PetscErrorCode ierr;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense = PETSC_FALSE;
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isdense);
        CHKERRQ(ierr);
        if (!isdense)
            SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE");
    }
#endif

    ierr = MatGetSize(A, &m, NULL);
    CHKERRQ(ierr);
    ierr = MatDenseGetArray(A, &_A);
    CHKERRQ(ierr);
    for (i = 0; i < m; i++) {
        for (j = i; j < m; j++) {
            PetscReal Aij, Aji, At;

            Aij = _A[i + j * m];
            Aji = _A[j + i * m];
            At = 0.5 * (Aij + Aji);

            _A[i + j * m] = At;
            _A[j + i * m] = At;
        }
    }
    ierr = MatDenseRestoreArray(A, &_A);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode lapackeigs(Mat R, Vec eigs) {
    PetscErrorCode ierr;
    PetscInt M;
    PetscScalar *_array, *_w;
    int dim, n, lda, info, lwork;
    double wkopt;
    double *work, *w, *a;

    PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
    {
        PetscBool isdense = PETSC_FALSE;
        ierr = PetscObjectTypeCompare((PetscObject)R, MATSEQDENSE, &isdense);
        CHKERRQ(ierr);
        if (!isdense)
            SETERRQ(PetscObjectComm((PetscObject)R), PETSC_ERR_SUP,
                    "Only valid for MatType MATSEQDENSE");
    }
#endif

    ierr = MatGetSize(R, &M, NULL);
    CHKERRQ(ierr);
    dim = (int)M;
    n = dim;
    lda = dim;

    ierr = MatDenseGetArray(R, &_array);
    CHKERRQ(ierr);
    a = (double*)_array;

    ierr = VecGetArray(eigs, &_w);
    CHKERRQ(ierr);
    w = (double*)_w;

    /* Query and allocate the optimal workspace */
    lwork = -1;
    FC_dsyev("Vectors", "Upper", &n, a, &lda, w, &wkopt, &lwork, &info);

    lwork = (int)wkopt;
    work = (double*)malloc(lwork * sizeof(double));
    memset(work, 0, lwork * sizeof(double));

    /* Solve eigenproblem */
    FC_dsyev("Vectors", "Upper", &n, a, &lda, w, work, &lwork, &info);

    ierr = MatDenseRestoreArray(R, &_array);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(eigs, &_w);
    CHKERRQ(ierr);
    free(work);
    PetscFunctionReturn(0);
}

PetscErrorCode RandEigsMax_3_InPlace(Mat A, PetscInt k, PetscInt o, PetscInt power_its,
                                     PetscRandom prand, Vec* _eigs, Mat* _V) {
    PetscErrorCode ierr;
    Vec eigs;
    Mat X, R, R_red;
    PetscInt it, M, m, i;
    PetscMPIInt commsize;
    MPI_Comm comm;

    PetscFunctionBegin;
    ierr = PetscObjectGetComm((PetscObject)A, &comm);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &commsize);
    CHKERRQ(ierr);

    ierr = MatGetSize(A, &M, NULL);
    CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &m, NULL);
    CHKERRQ(ierr);
    if (!*_eigs) {
        ierr = VecCreate(PETSC_COMM_SELF, &eigs);
        CHKERRQ(ierr);
        ierr = VecSetSizes(eigs, PETSC_DECIDE, k);
        CHKERRQ(ierr);
        ierr = VecSetUp(eigs);
        CHKERRQ(ierr);
        *_eigs = eigs;
    } else {
        eigs = *_eigs;
    }

    ierr = MatCreate(comm, &X);
    CHKERRQ(ierr);
    ierr = MatSetSizes(X, m, PETSC_DECIDE, M, (k + o));
    CHKERRQ(ierr);
    ierr = MatSetType(X, MATDENSE);
    CHKERRQ(ierr);
    ierr = MatSetUp(X);
    CHKERRQ(ierr);

    ierr = MatSetRandom(X, prand);
    CHKERRQ(ierr);

    ierr = MatMatMult_MatDense_InPlace(A, X);
    CHKERRQ(ierr); /* aij/baij * dense */

    if (power_its == 0) {
        ierr = MatDenseOrthonormalize(X);
        CHKERRQ(ierr);
    } else {
        ierr = MatMatMult_MatDense_InPlace(A, X);
        CHKERRQ(ierr); /* aij/baij * dense */
        ierr = MatDenseOrthonormalize(X);
        CHKERRQ(ierr);

        for (it = 1; it < power_its; it++) {
            ierr = MatMatMult_MatDense_InPlace(A, X);
            CHKERRQ(ierr); /* aij/baij * dense */
            ierr = MatDenseOrthonormalize(X);
            CHKERRQ(ierr);
        }
    }

    ierr = MatPtAP_MatDense(A, X, MAT_INITIAL_MATRIX, &R);
    CHKERRQ(ierr);
    if (commsize > 1) {
        ierr = MatCreateRedundantMatrix(R, commsize, PETSC_COMM_SELF, MAT_INITIAL_MATRIX, &R_red);
        CHKERRQ(ierr);
    } else {
        ierr = PetscObjectReference((PetscObject)R);
        CHKERRQ(ierr);
        R_red = R;
    }
    ierr = MatDenseSymmetrizeSEQ(R_red);
    CHKERRQ(ierr);
    // ierr = MatView(R_red,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

    //(d, V) = eigh(R)
    {
        Vec eigs_ko;
        const PetscScalar* _eko;
        PetscScalar* _e;

        ierr = VecCreate(PETSC_COMM_SELF, &eigs_ko);
        CHKERRQ(ierr);
        ierr = VecSetSizes(eigs_ko, PETSC_DECIDE, k + o);
        CHKERRQ(ierr);
        ierr = VecSetUp(eigs_ko);
        CHKERRQ(ierr);

        ierr = lapackeigs(R_red, eigs_ko);
        CHKERRQ(ierr);

        ierr = VecGetArrayRead(eigs_ko, &_eko);
        CHKERRQ(ierr);
        ierr = VecGetArray(eigs, &_e);
        CHKERRQ(ierr);
        for (i = 0; i < k; i++) {
            _e[i] = _eko[i + o];
        }
        ierr = VecRestoreArray(eigs, &_e);
        CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(eigs_ko, &_eko);
        CHKERRQ(ierr);
        ierr = VecDestroy(&eigs_ko);
        CHKERRQ(ierr);
    }

    if (_V) {
        Mat Q, V;
        PetscInt j;
        Vec Rj, Vj;

        if (!*_V) {
            ierr = MatCreate(comm, &V);
            CHKERRQ(ierr);
            ierr = MatSetSizes(V, m, PETSC_DECIDE, M, k);
            CHKERRQ(ierr);
            ierr = MatSetType(V, MATDENSE);
            CHKERRQ(ierr);
            ierr = MatSetUp(V);
            CHKERRQ(ierr);
            *_V = V;
        } else {
            V = *_V;
        }

        // V = np.matmul(Q, R)
        Q = X;

        ierr = _MatDenseCreateEmptyColumnVec(R_red, &Rj);
        CHKERRQ(ierr);
        ierr = _MatDenseCreateEmptyColumnVec(V, &Vj);
        CHKERRQ(ierr);
        for (j = o; j < k + o; j++) {
            ierr = _MatDenseSetColumnVec(R_red, j, Rj);
            CHKERRQ(ierr);
            ierr = _MatDenseSetColumnVec(V, j - o, Vj);
            CHKERRQ(ierr);
            ierr = MatMultRedundant_MatDenseVecSeq(Q, Rj, Vj);
            CHKERRQ(ierr);
            ierr = _MatDenseResetColumnVec(V, j - o, Vj);
            CHKERRQ(ierr);
            ierr = _MatDenseResetColumnVec(R_red, j, Rj);
            CHKERRQ(ierr);
        }
        ierr = VecDestroy(&Vj);
        CHKERRQ(ierr);
        ierr = VecDestroy(&Rj);
        CHKERRQ(ierr);

        ierr = MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        ierr = MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
    }

    ierr = MatDestroy(&R_red);
    CHKERRQ(ierr);
    ierr = MatDestroy(&R);
    CHKERRQ(ierr);
    ierr = MatDestroy(&X);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode RandEigsMin_3_InPlace(KSP ksp, PetscInt k, PetscInt o, PetscInt power_its,
                                     PetscRandom prand, Vec* _eigs, Mat* _V) {
    PetscErrorCode ierr;
    Vec eigs, eigs_ko;
    Mat A, X, R, R_red;
    PetscInt it, M, m, i;
    MPI_Comm comm;
    PetscMPIInt commsize;
    PetscBool flg = PETSC_FALSE;

    PetscFunctionBegin;
    ierr = KSPGetOperators(ksp, &A, NULL);
    CHKERRQ(ierr);
    ierr = MatGetSize(A, &M, NULL);
    CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &m, NULL);
    CHKERRQ(ierr);
    if (!*_eigs) {
        ierr = VecCreate(PETSC_COMM_SELF, &eigs);
        CHKERRQ(ierr);
        ierr = VecSetSizes(eigs, PETSC_DECIDE, k);
        CHKERRQ(ierr);
        ierr = VecSetUp(eigs);
        CHKERRQ(ierr);
        *_eigs = eigs;
    } else {
        eigs = *_eigs;
    }

    ierr = PetscObjectGetComm((PetscObject)ksp, &comm);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &commsize);
    CHKERRQ(ierr);

    ierr = MatCreate(comm, &X);
    CHKERRQ(ierr);
    ierr = MatSetSizes(X, m, PETSC_DECIDE, M, (k + o));
    CHKERRQ(ierr);
    ierr = MatSetType(X, MATDENSE);
    CHKERRQ(ierr);
    ierr = MatSetUp(X);
    CHKERRQ(ierr);

    ierr = MatSetRandom(X, prand);
    CHKERRQ(ierr);

    ierr = KSPMatMult_MatDense_InPlace(ksp, X, flg);
    CHKERRQ(ierr);

    if (power_its == 0) {
        ierr = MatDenseOrthonormalize(X);
        CHKERRQ(ierr);
    } else {
        ierr = KSPMatMult_MatDense_InPlace(ksp, X, flg);
        CHKERRQ(ierr);
        ierr = MatDenseOrthonormalize(X);
        CHKERRQ(ierr);

        for (it = 1; it < power_its; it++) {
            ierr = KSPMatMult_MatDense_InPlace(ksp, X, flg);
            CHKERRQ(ierr);
            ierr = MatDenseOrthonormalize(X);
            CHKERRQ(ierr);
        }
    }

    ierr = MatPtInvAP_MatDense(ksp, X, MAT_INITIAL_MATRIX, &R);
    CHKERRQ(ierr);
    if (commsize > 1) {
        ierr = MatCreateRedundantMatrix(R, commsize, PETSC_COMM_SELF, MAT_INITIAL_MATRIX, &R_red);
        CHKERRQ(ierr);
    } else {
        ierr = PetscObjectReference((PetscObject)R);
        CHKERRQ(ierr);
        R_red = R;
    }
    ierr = MatDenseSymmetrizeSEQ(R_red);
    CHKERRQ(ierr);

    ierr = MatCreateVecs(R_red, NULL, &eigs_ko);
    CHKERRQ(ierr);

    //(d, V) = eigh(R)
    {
        PetscScalar *_eko, *_e;

        ierr = lapackeigs(R_red, eigs_ko);
        CHKERRQ(ierr);

        /* invert stage of shift-and-invert */
        ierr = VecReciprocal(eigs_ko);
        CHKERRQ(ierr);

        ierr = VecGetArray(eigs_ko, &_eko);
        CHKERRQ(ierr);
        ierr = VecGetArray(eigs, &_e);
        CHKERRQ(ierr);

        /* reverse the order of the eigenvalues */
        {
            PetscInt first, last, center;
            PetscScalar tmp;

            center = (k + o) / 2;

            for (i = 0; i < center; i++) {
                first = i;
                last = (k + o) - 1 - i;

                tmp = _eko[last];
                _eko[last] = _eko[first];
                _eko[first] = tmp;
            }
        }

        /* keep the first k terms */
        for (i = 0; i < k; i++) {
            _e[i] = _eko[i];
        }

        /* zero the last o terms */
        for (i = k; i < k + o; i++) {
            _eko[i] = 0.0;
        }

        ierr = VecRestoreArray(eigs, &_e);
        CHKERRQ(ierr);
        ierr = VecRestoreArray(eigs_ko, &_eko);
        CHKERRQ(ierr);
        ierr = VecDestroy(&eigs_ko);
        CHKERRQ(ierr);

        /* reverse order of eigenvectors in R */
        {
            PetscInt j, first, last, center;
            Vec tmp, Rj, Rn;

            center = (k + o) / 2;

            ierr = MatCreateVecs(R_red, NULL, &tmp);
            CHKERRQ(ierr);
            ierr = _MatDenseCreateEmptyColumnVec(R_red, &Rj);
            CHKERRQ(ierr);
            ierr = _MatDenseCreateEmptyColumnVec(R_red, &Rn);
            CHKERRQ(ierr);

            for (j = 0; j < center; j++) {
                first = j;
                last = (k + o) - 1 - j;

                ierr = _MatDenseSetColumnVec(R_red, j, Rj);
                CHKERRQ(ierr);
                ierr = _MatDenseSetColumnVec(R_red, last, Rn);
                CHKERRQ(ierr);

                ierr = VecCopy(Rn, tmp);
                CHKERRQ(ierr);
                ierr = VecCopy(Rj, Rn);
                CHKERRQ(ierr);
                ierr = VecCopy(tmp, Rj);
                CHKERRQ(ierr);

                ierr = _MatDenseResetColumnVec(R_red, j, Rj);
                CHKERRQ(ierr);
                ierr = _MatDenseResetColumnVec(R_red, last, Rn);
                CHKERRQ(ierr);
            }

            ierr = VecDestroy(&tmp);
            CHKERRQ(ierr);
            ierr = VecDestroy(&Rj);
            CHKERRQ(ierr);
            ierr = VecDestroy(&Rn);
            CHKERRQ(ierr);
        }
    }

    /*
     X = Q R
     V = [Q R] P_{col_perm}
     */

    if (_V) {
        Mat Q, V;
        PetscInt j;
        Vec Rj, Vj;

        if (!*_V) {
            ierr = MatCreate(comm, &V);
            CHKERRQ(ierr);
            ierr = MatSetSizes(V, m, PETSC_DECIDE, M, k);
            CHKERRQ(ierr);
            ierr = MatSetType(V, MATDENSE);
            CHKERRQ(ierr);
            ierr = MatSetUp(V);
            CHKERRQ(ierr);
            *_V = V;
        } else {
            V = *_V;
        }

        // V = np.matmul(Q, R)
        Q = X;

        ierr = _MatDenseCreateEmptyColumnVec(R_red, &Rj);
        CHKERRQ(ierr);
        ierr = _MatDenseCreateEmptyColumnVec(V, &Vj);
        CHKERRQ(ierr);
        for (j = 0; j < k; j++) {
            ierr = _MatDenseSetColumnVec(R_red, j, Rj);
            CHKERRQ(ierr);
            ierr = _MatDenseSetColumnVec(V, j, Vj);
            CHKERRQ(ierr);
            ierr = MatMultRedundant_MatDenseVecSeq(Q, Rj, Vj);
            CHKERRQ(ierr);
            ierr = _MatDenseResetColumnVec(V, j, Vj);
            CHKERRQ(ierr);
            ierr = _MatDenseResetColumnVec(R_red, j, Rj);
            CHKERRQ(ierr);
        }
        ierr = VecDestroy(&Vj);
        CHKERRQ(ierr);
        ierr = VecDestroy(&Rj);
        CHKERRQ(ierr);

        ierr = MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        ierr = MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
    }

    ierr = VecDestroy(&eigs_ko);
    CHKERRQ(ierr);
    ierr = MatDestroy(&R);
    CHKERRQ(ierr);
    ierr = MatDestroy(&R_red);
    CHKERRQ(ierr);
    ierr = MatDestroy(&X);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode RandEigsMax(Mat A, PetscInt k, PetscInt o, PetscInt pits, PetscRandom prand,
                           Vec* _eigs, Mat* _V) {
    PetscErrorCode ierr;
    MPI_Comm comm;
    PetscMPIInt commsize;
    PetscInt M, N;

    PetscFunctionBegin;
    comm = PetscObjectComm((PetscObject)A);
    ierr = MPI_Comm_size(comm, &commsize);
    CHKERRQ(ierr);
    ierr = MatGetSize(A, &M, &N);
    CHKERRQ(ierr);
    if (M != N)
        SETERRQ(comm, PETSC_ERR_SUP,
                "Only valid for square matrices, found M = %" PetscInt_FMT ", N = %" PetscInt_FMT
                "\n",
                M, N);
    if (k + o > M)
        SETERRQ(comm, PETSC_ERR_SUP,
                "Random matrix has %" PetscInt_FMT " + %" PetscInt_FMT " = %" PetscInt_FMT
                " columns. Max num. columns is %" PetscInt_FMT "\n",
                k, o, k + o, M);
    ierr = RandEigsMax_3_InPlace(A, k, o, pits, prand, _eigs, _V);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode RandEigsMin(KSP ksp, PetscInt k, PetscInt o, PetscInt pits, PetscRandom prand,
                           Vec* _eigs, Mat* _V) {
    PetscErrorCode ierr;
    MPI_Comm comm;
    PetscMPIInt commsize;
    Mat A;
    PetscInt M, N;

    PetscFunctionBegin;
    ierr = KSPGetOperators(ksp, &A, NULL);
    CHKERRQ(ierr);
    comm = PetscObjectComm((PetscObject)A);
    ierr = MPI_Comm_size(comm, &commsize);
    CHKERRQ(ierr);
    ierr = MatGetSize(A, &M, &N);
    CHKERRQ(ierr);
    if (M != N)
        SETERRQ(comm, PETSC_ERR_SUP,
                "Only valid for square matrices, found M = %" PetscInt_FMT ", N = %" PetscInt_FMT
                "\n",
                M, N);
    if (k + o > M)
        SETERRQ(comm, PETSC_ERR_SUP,
                "Random matrix has %" PetscInt_FMT " + %" PetscInt_FMT " = %" PetscInt_FMT
                " columns. Max num. columns is %" PetscInt_FMT "\n",
                k, o, k + o, M);
    ierr = RandEigsMin_3_InPlace(ksp, k, o, pits, prand, _eigs, _V);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
