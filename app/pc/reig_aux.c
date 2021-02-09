#include "reig_aux.h"
#include "common/lapack.h"

PetscErrorCode MatOrthogonalizeModifiedGramSchmidt(PetscInt n,Vec X[])
{
  PetscErrorCode ierr;
  Vec            q,v;
  PetscInt       j,k;
  PetscReal      nrm,dot;
  
  for (j=0; j<n; j++) {
    q = X[j];
    ierr = VecNorm(q,NORM_2,&nrm);CHKERRQ(ierr);
    ierr = VecScale(q,1.0/nrm);CHKERRQ(ierr);
    for (k=j+1; k<n; k++) {
      v = X[k];
      ierr = VecDot(q,v,&dot);CHKERRQ(ierr);
      ierr = VecAXPY(v,-dot,q);CHKERRQ(ierr);
    }
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode MatOrthogonalizeMatDense(Mat Q)
{
  PetscErrorCode ierr;
  PetscInt       j,M,m,N,bs;
  Vec            *q;
  PetscScalar    *array;
  
  /* Create array of vectors */
  ierr = MatGetSize(Q,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Q,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetBlockSize(Q,&bs);CHKERRQ(ierr);
  
  ierr = MatDenseGetArray(Q,&array);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(N,&q);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    PetscScalar *array_j = array + j * m;
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,bs,m,M,(const PetscScalar*)array_j,&q[j]);CHKERRQ(ierr);
  }

  ierr = MatDenseRestoreArray(Q,&array);CHKERRQ(ierr);

  ierr = MatOrthogonalizeModifiedGramSchmidt(N,q);CHKERRQ(ierr);
  
  for (j=0; j<N; j++) {
    ierr = VecDestroy(&q[j]);CHKERRQ(ierr);
  }
  ierr = PetscFree(q);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseCopyFast(Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  Mat            lA;
  PetscScalar    *_a,*_b;
  
  ierr = MatDenseGetLocalMatrix(A,&lA);CHKERRQ(ierr);
  ierr = MatGetSize(lA,&m,&n);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&_a);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&_b);CHKERRQ(ierr);
  ierr = PetscMemcpy(_b,_a,sizeof(PetscScalar)*m*n);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&_a);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(A,&_b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _MatDenseCreateEmptyColumnVec(Mat A,Vec *v)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n,bs;
  
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),bs,m,M,NULL,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _MatDenseSetColumnVec(Mat A,PetscInt j,Vec v)
{
  PetscErrorCode ierr;
  PetscInt       lda;
  PetscScalar    *array;
  
  ierr = MatDenseGetLDA(A,&lda);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(v,array + (size_t)j * (size_t)lda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _MatDenseResetColumnVec(Mat A,PetscInt j,Vec v)
{
  PetscErrorCode ierr;
  ierr = VecResetArray(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetRandomConsistent(Mat Q)
{
  PetscErrorCode    ierr;
  Vec               q;
  PetscInt          M,N,j,i,start,end;
  PetscRandom       randObj;
  const PetscScalar *_q;

  ierr = MatZeroEntries(Q);CHKERRQ(ierr);
  ierr = MatGetSize(Q,&M,&N);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&q);CHKERRQ(ierr);
  ierr = VecSetSizes(q,PETSC_DECIDE,M);CHKERRQ(ierr);
  ierr = VecSetType(q,VECSEQ);CHKERRQ(ierr);
  
  ierr = MatGetOwnershipRange(Q,&start,&end);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&randObj);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randObj);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    ierr = VecSetRandom(q,randObj);CHKERRQ(ierr);
    ierr = VecGetArrayRead(q,&_q);CHKERRQ(ierr);
    for (i=start; i<end; i++) {
      ierr = MatSetValue(Q,i,j,_q[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(q,&_q);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&randObj);CHKERRQ(ierr);
  ierr = VecDestroy(&q);CHKERRQ(ierr);
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
PetscErrorCode MatMultRedundant_MatDenseVecSeq(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat            lA;
  Vec            ly;
  PetscInt       m,bs;
  PetscScalar    *_y;
  
  ierr = VecGetLocalSize(y,&m);CHKERRQ(ierr);
  ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
  ierr = VecGetArray(y,&_y);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,m,_y,&ly);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&_y);CHKERRQ(ierr);
  ierr = MatDenseGetLocalMatrix(A,&lA);CHKERRQ(ierr);
  ierr = MatMult(lA,x,ly);CHKERRQ(ierr);
  ierr = VecDestroy(&ly);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Solve
    A Y[i] = X[i]
  for all columns i in X
*/
PetscErrorCode MatDenseSolveKSP(KSP ksp,Mat X,MatReuse scall,Mat *_Y)
{
  PetscErrorCode ierr;
  PetscInt       j,N;
  Vec            Xj,Yj;
  Mat            Y;
  
  if (scall == MAT_INITIAL_MATRIX) {
    PetscInt m,n,M,N;
    
    ierr = MatGetSize(X,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(X,&m,&n);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)X),&Y);CHKERRQ(ierr);
    ierr = MatSetSizes(Y,m,n,M,N);CHKERRQ(ierr);
    ierr = MatSetType(Y,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(Y);CHKERRQ(ierr);
    *_Y = Y;
  } else if (scall == MAT_REUSE_MATRIX) {
    Y = *_Y;
  } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Only support for MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX");
  
  ierr = MatZeroEntries(Y);CHKERRQ(ierr);
  
  ierr = _MatDenseCreateEmptyColumnVec(X,&Xj);CHKERRQ(ierr);
  ierr = _MatDenseCreateEmptyColumnVec(Y,&Yj);CHKERRQ(ierr);
  
  ierr = MatGetSize(X,NULL,&N);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    ierr = _MatDenseSetColumnVec(X,j,Xj);CHKERRQ(ierr);
    ierr = _MatDenseSetColumnVec(Y,j,Yj);CHKERRQ(ierr);
    //ierr = MatDenseGetColumnVec(X,j,&Xj);CHKERRQ(ierr);
    //ierr = MatDenseGetColumnVec(Y,j,&Yj);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,Xj,Yj);CHKERRQ(ierr);
    //ierr = MatDenseRestoreColumnVec(Y,j,&Yj);CHKERRQ(ierr);
    //ierr = MatDenseRestoreColumnVec(X,j,&Xj);CHKERRQ(ierr);
    ierr = _MatDenseResetColumnVec(Y,j,Yj);CHKERRQ(ierr);
    ierr = _MatDenseResetColumnVec(X,j,Xj);CHKERRQ(ierr);
  }
  
  ierr = VecDestroy(&Xj);CHKERRQ(ierr);
  ierr = VecDestroy(&Yj);CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode lapackeigs(Mat R,Vec eigs)
{
  PetscErrorCode ierr;
  PetscInt       M;
  PetscScalar    *_array,*_w;
  int            dim,n,lda,info,lwork;
  double         wkopt;
  double         *work,*w,*a;
  
  ierr = MatGetSize(R,&M,NULL);CHKERRQ(ierr);
  dim = (int)M;
  n = dim;
  lda = dim;
  
  ierr = MatDenseGetArray(R,&_array);CHKERRQ(ierr);
  a = (double*)_array;
  
  ierr = VecGetArray(eigs,&_w);CHKERRQ(ierr);
  w = (double*)_w;
  
  /* Query and allocate the optimal workspace */
  lwork = -1;
  FC_dsyev( "Vectors", "Upper", &n, a, &lda, w, &wkopt, &lwork, &info );
  
  lwork = (int)wkopt;
  work = (double*)malloc(lwork*sizeof(double));
  memset(work,0,lwork*sizeof(double));
  
  /* Solve eigenproblem */
  FC_dsyev( "Vectors", "Upper", &n, a, &lda, w, work, &lwork, &info );
  
  ierr = MatDenseRestoreArray(R,&_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(eigs,&_w);CHKERRQ(ierr);
  free(work);
  PetscFunctionReturn(0);
}

PetscErrorCode RandEigsMax_SEQ_2(Mat A,PetscInt k,PetscInt o,PetscInt power_its,PetscRandom prand,Vec *_eigs,Mat *_V)
{
  PetscErrorCode ierr;
  Vec            eigs;
  Mat            S,Q,V,R;
  PetscInt       it,M,m,i;
  
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  if (!*_eigs) {
    ierr = VecCreate(PETSC_COMM_SELF,&eigs);CHKERRQ(ierr);
    ierr = VecSetSizes(eigs,PETSC_DECIDE,k);CHKERRQ(ierr);
    ierr = VecSetUp(eigs);CHKERRQ(ierr);
    *_eigs = eigs;
  } else {
    eigs = *_eigs;
  }
  if (!*_V) {
    ierr = MatCreate(PETSC_COMM_SELF,&V);CHKERRQ(ierr);
    ierr = MatSetSizes(V,m,PETSC_DECIDE,M,k);CHKERRQ(ierr);
    ierr = MatSetType(V,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(V);CHKERRQ(ierr);
    *_V = V;
  } else {
    V = *_V;
  }
  
  ierr = MatCreate(PETSC_COMM_SELF,&S);CHKERRQ(ierr);
  ierr = MatSetSizes(S,m,PETSC_DECIDE,M,(k+o));CHKERRQ(ierr);
  ierr = MatSetType(S,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  
  ierr = MatSetRandom(S,prand);CHKERRQ(ierr);
  
  ierr = MatMatMult(A,S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Q);CHKERRQ(ierr); /* aij/baij * dense */
  
  if (power_its == 0) {
    ierr = MatOrthogonalizeMatDense(Q);CHKERRQ(ierr);
  } else {
    Mat AQ;
    
    ierr = MatMatMult(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AQ);CHKERRQ(ierr); /* aij/baij * dense */
    ierr = MatOrthogonalizeMatDense(AQ);CHKERRQ(ierr);
    ierr = MatCopy(AQ,Q,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    
    for (it=1; it<power_its; it++) {
      ierr = MatMatMult(A,Q,MAT_REUSE_MATRIX,PETSC_DEFAULT,&AQ);CHKERRQ(ierr); /* aij/baij * dense */
      ierr = MatOrthogonalizeMatDense(AQ);CHKERRQ(ierr);
      ierr = MatCopy(AQ,Q,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    
    ierr = MatDestroy(&AQ);CHKERRQ(ierr);
  }
  
  ierr = MatPtAP(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R);
  
  //(d, V) = eigh(R)
  {
    Vec               eigs_ko;
    const PetscScalar *_eko;
    PetscScalar       *_e;
    
    ierr = VecCreate(PETSC_COMM_SELF,&eigs_ko);CHKERRQ(ierr);
    ierr = VecSetSizes(eigs_ko,PETSC_DECIDE,k+o);CHKERRQ(ierr);
    ierr = VecSetUp(eigs_ko);CHKERRQ(ierr);
    
    ierr = lapackeigs(R,eigs_ko);CHKERRQ(ierr);
    
    ierr = VecGetArrayRead(eigs_ko,&_eko);CHKERRQ(ierr);
    ierr = VecGetArray(eigs,&_e);CHKERRQ(ierr);
    for (i=0; i<k; i++) {
      _e[i] = _eko[i+o];
    }
    ierr = VecRestoreArray(eigs,&_e);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(eigs_ko,&_eko);CHKERRQ(ierr);
    ierr = VecDestroy(&eigs_ko);CHKERRQ(ierr);
  }
  
  //V = np.matmul(Q, V)
  ierr = MatMatMult(Q,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&S);CHKERRQ(ierr); /* dense * seqdesnse */
  {
    const PetscScalar *_Vko;
    PetscScalar       *_V;
    PetscInt          offset;
    
    ierr = MatDenseGetArrayRead(S,&_Vko);CHKERRQ(ierr);
    ierr = MatDenseGetArray(V,&_V);CHKERRQ(ierr);
    offset = o * M;
    ierr = PetscMemcpy(_V,_Vko + offset,sizeof(PetscScalar)*M*k);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(V,&_V);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(S,&_Vko);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(V,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(V,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode RandEigsMax_MPI_2(Mat A,PetscInt k,PetscInt o,PetscInt power_its,PetscRandom prand,Vec *_eigs,Mat *_V)
{
  PetscErrorCode ierr;
  Vec            eigs;
  Mat            S,Q,V,R,R_red;
  PetscInt       it,M,m,i;
  PetscMPIInt    commsize;
  MPI_Comm       comm;
  
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&commsize);CHKERRQ(ierr);
  
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  if (!*_eigs) {
    ierr = VecCreate(PETSC_COMM_SELF,&eigs);CHKERRQ(ierr);
    ierr = VecSetSizes(eigs,PETSC_DECIDE,k);CHKERRQ(ierr);
    ierr = VecSetUp(eigs);CHKERRQ(ierr);
    *_eigs = eigs;
  } else {
    eigs = *_eigs;
  }
  if (!*_V) {
    ierr = MatCreate(comm,&V);CHKERRQ(ierr);
    ierr = MatSetSizes(V,m,PETSC_DECIDE,M,k);CHKERRQ(ierr);
    ierr = MatSetType(V,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(V);CHKERRQ(ierr);
    *_V = V;
  } else {
    V = *_V;
  }
  
  ierr = MatCreate(comm,&S);CHKERRQ(ierr);
  ierr = MatSetSizes(S,m,PETSC_DECIDE,M,(k+o));CHKERRQ(ierr);
  ierr = MatSetType(S,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  
  ierr = MatSetRandom(S,prand);CHKERRQ(ierr);
  
  ierr = MatMatMult(A,S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Q);CHKERRQ(ierr); /* aij/baij * dense */
  
  if (power_its == 0) {
    ierr = MatOrthogonalizeMatDense(Q);CHKERRQ(ierr);
  } else {
    Mat AQ;
    
    ierr = MatMatMult(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AQ);CHKERRQ(ierr); /* aij/baij * dense */
    ierr = MatOrthogonalizeMatDense(AQ);CHKERRQ(ierr);
    //ierr = MatCopy(AQ,Q,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDenseCopyFast(AQ,Q);CHKERRQ(ierr);
    
    for (it=1; it<power_its; it++) {
      ierr = MatMatMult(A,Q,MAT_REUSE_MATRIX,PETSC_DEFAULT,&AQ);CHKERRQ(ierr); /* aij/baij * dense */
      ierr = MatOrthogonalizeMatDense(AQ);CHKERRQ(ierr);
      //ierr = MatCopy(AQ,Q,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDenseCopyFast(AQ,Q);CHKERRQ(ierr);
    }
    
    ierr = MatDestroy(&AQ);CHKERRQ(ierr);
  }
  
  ierr = MatPtAP(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R);
  ierr = MatCreateRedundantMatrix(R,commsize,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&R_red);CHKERRQ(ierr);
  
  //ierr = MatView(R_red,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  
  //(d, V) = eigh(R)
  {
    Vec               eigs_ko;
    const PetscScalar *_eko;
    PetscScalar       *_e;
    
    ierr = VecCreate(PETSC_COMM_SELF,&eigs_ko);CHKERRQ(ierr);
    ierr = VecSetSizes(eigs_ko,PETSC_DECIDE,k+o);CHKERRQ(ierr);
    ierr = VecSetUp(eigs_ko);CHKERRQ(ierr);
    
    ierr = lapackeigs(R_red,eigs_ko);CHKERRQ(ierr);
    
    ierr = VecGetArrayRead(eigs_ko,&_eko);CHKERRQ(ierr);
    ierr = VecGetArray(eigs,&_e);CHKERRQ(ierr);
    for (i=0; i<k; i++) {
      _e[i] = _eko[i+o];
    }
    ierr = VecRestoreArray(eigs,&_e);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(eigs_ko,&_eko);CHKERRQ(ierr);
    ierr = VecDestroy(&eigs_ko);CHKERRQ(ierr);
  }
  
  //V = np.matmul(Q, V)
  //ierr = MatMatMult(Q,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&V);CHKERRQ(ierr); /* dense * seqdesnse */
  {
    Mat Q_red,V_red;
    PetscScalar *_array_Q,*_array_V;
    
    ierr = MatDenseGetArray(Q,&_array_Q);CHKERRQ(ierr);
    ierr = MatDenseGetArray(S,&_array_V);CHKERRQ(ierr);
    
    ierr = MatCreate(PETSC_COMM_SELF,&Q_red);CHKERRQ(ierr);
    ierr = MatSetSizes(Q_red,m,(k+o),m,(k+o));CHKERRQ(ierr);
    ierr = MatSetType(Q_red,MATDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(Q_red,_array_Q);CHKERRQ(ierr);
    ierr = MatSetUp(Q_red);CHKERRQ(ierr);
    
    ierr = MatCreate(PETSC_COMM_SELF,&V_red);CHKERRQ(ierr);
    ierr = MatSetSizes(V_red,m,(k+o),m,(k+o));CHKERRQ(ierr);
    ierr = MatSetType(V_red,MATDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(V_red,_array_V);CHKERRQ(ierr);
    ierr = MatSetUp(V_red);CHKERRQ(ierr);
    
    ierr = MatMatMult(Q_red,R_red,MAT_REUSE_MATRIX,PETSC_DEFAULT,&V_red);CHKERRQ(ierr);
    
    ierr = MatDestroy(&V_red);CHKERRQ(ierr);
    ierr = MatDestroy(&Q_red);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(Q,&_array_Q);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(S,&_array_V);CHKERRQ(ierr);
  }
  
  
  {
    const PetscScalar *_Vko;
    PetscScalar       *_V;
    PetscInt          offset;
    Mat               lS,lV;
    
    ierr = MatDenseGetLocalMatrix(S,&lS);CHKERRQ(ierr);
    ierr = MatDenseGetLocalMatrix(V,&lV);CHKERRQ(ierr);
    
    ierr = MatDenseGetArrayRead(lS,&_Vko);CHKERRQ(ierr);
    ierr = MatDenseGetArray(lV,&_V);CHKERRQ(ierr);
    offset = o * m;
    ierr = PetscMemcpy(_V,_Vko + offset,sizeof(PetscScalar)*m*k);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(lV,&_V);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(lS,&_Vko);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(V,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(V,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = MatDestroy(&R_red);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode RandEigsMin_SEQ_2(KSP ksp,PetscInt k,PetscInt o,PetscInt power_its,PetscRandom prand,Vec *_eigs,Mat *_V)
{
  PetscErrorCode ierr;
  Vec            eigs;
  Mat            A,S,Q,V,R;
  PetscInt       it,M,m,i;
  
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  if (!*_eigs) {
    ierr = VecCreate(PETSC_COMM_SELF,&eigs);CHKERRQ(ierr);
    ierr = VecSetSizes(eigs,PETSC_DECIDE,k);CHKERRQ(ierr);
    ierr = VecSetUp(eigs);CHKERRQ(ierr);
    *_eigs = eigs;
  } else {
    eigs = *_eigs;
  }
  if (!*_V) {
    ierr = MatCreate(PETSC_COMM_SELF,&V);CHKERRQ(ierr);
    ierr = MatSetSizes(V,m,PETSC_DECIDE,M,k);CHKERRQ(ierr);
    ierr = MatSetType(V,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(V);CHKERRQ(ierr);
    *_V = V;
  } else {
    V = *_V;
  }
  
  ierr = MatCreate(PETSC_COMM_SELF,&S);CHKERRQ(ierr);
  ierr = MatSetSizes(S,m,PETSC_DECIDE,M,(k+o));CHKERRQ(ierr);
  ierr = MatSetType(S,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  
  ierr = MatSetRandom(S,prand);CHKERRQ(ierr);
  
  //ierr = MatMatMult(A,S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Q);CHKERRQ(ierr); /* aij/baij * dense */
  ierr = MatDenseSolveKSP(ksp,S,MAT_INITIAL_MATRIX,&Q);CHKERRQ(ierr);
  
  if (power_its == 0) {
    ierr = MatOrthogonalizeMatDense(Q);CHKERRQ(ierr);
  } else {
    Mat AQ;
    
    //ierr = MatMatMult(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AQ);CHKERRQ(ierr); /* aij/baij * dense */
    ierr = MatDenseSolveKSP(ksp,Q,MAT_INITIAL_MATRIX,&AQ);CHKERRQ(ierr);
    ierr = MatOrthogonalizeMatDense(AQ);CHKERRQ(ierr);
    ierr = MatCopy(AQ,Q,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    
    for (it=1; it<power_its; it++) {
      //ierr = MatMatMult(A,Q,MAT_REUSE_MATRIX,PETSC_DEFAULT,&AQ);CHKERRQ(ierr); /* aij/baij * dense */
      ierr = MatDenseSolveKSP(ksp,Q,MAT_REUSE_MATRIX,&AQ);CHKERRQ(ierr);
      ierr = MatOrthogonalizeMatDense(AQ);CHKERRQ(ierr);
      ierr = MatCopy(AQ,Q,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    
    ierr = MatDestroy(&AQ);CHKERRQ(ierr);
  }
  
  //ierr = MatPtAP(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R);
  {
    Mat AQ;
    
    ierr = MatDenseSolveKSP(ksp,Q,MAT_INITIAL_MATRIX,&AQ);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(Q,AQ,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R);CHKERRQ(ierr);
    
    ierr = MatDestroy(&AQ);CHKERRQ(ierr);
  }
  
  //(d, V) = eigh(R)
  {
    Vec               eigs_ko;
    PetscScalar       *_eko,*_e;
    
    ierr = VecCreate(PETSC_COMM_SELF,&eigs_ko);CHKERRQ(ierr);
    ierr = VecSetSizes(eigs_ko,PETSC_DECIDE,k+o);CHKERRQ(ierr);
    ierr = VecSetUp(eigs_ko);CHKERRQ(ierr);
    
    ierr = lapackeigs(R,eigs_ko);CHKERRQ(ierr);
    
    /* invert stage of shift-and-invert */
    ierr = VecReciprocal(eigs_ko);CHKERRQ(ierr);
    
    ierr = VecGetArray(eigs_ko,&_eko);CHKERRQ(ierr);
    ierr = VecGetArray(eigs,&_e);CHKERRQ(ierr);
    
    /* reverse the order of the full eigenvector */
    {
      PetscInt    first,last,center;
      PetscScalar tmp;
      
      center = (k+o)/2;
      
      for (i=0; i<center; i++) {
        first = i;
        last = (k+o) - 1 - i;
        
        tmp = _eko[last];
        _eko[last] = _eko[first];
        _eko[first] = tmp;
      }
    }
    
    /* keep the first k terms */
    for (i=0; i<k; i++) {
      _e[i] = _eko[i];
    }
    ierr = VecRestoreArray(eigs,&_e);CHKERRQ(ierr);
    ierr = VecRestoreArray(eigs_ko,&_eko);CHKERRQ(ierr);
    ierr = VecDestroy(&eigs_ko);CHKERRQ(ierr);
  }

  
  //V = np.matmul(Q, V)
  ierr = MatMatMult(Q,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&S);CHKERRQ(ierr); /* dense * seqdesnse */
  
  /* reverse order of eigenvectors */
  {
    PetscInt j,first,last,center;
    Vec      tmp,Vj,Vn;
    
    center = (k+o)/2;
    
    ierr = MatCreateVecs(S,NULL,&tmp);CHKERRQ(ierr);
    ierr = _MatDenseCreateEmptyColumnVec(S,&Vj);CHKERRQ(ierr);
    ierr = _MatDenseCreateEmptyColumnVec(S,&Vn);CHKERRQ(ierr);
    
    for (j=0; j<center; j++) {
      first = j;
      last = (k+o) - 1 - j;
      
      ierr = _MatDenseSetColumnVec(S,j,Vj);CHKERRQ(ierr);
      ierr = _MatDenseSetColumnVec(S,last,Vn);CHKERRQ(ierr);
      
      //ierr = MatDenseGetColumnVec(S,j,&Vj);CHKERRQ(ierr);
      //ierr = MatDenseGetColumnVec(S,last,&Vn);CHKERRQ(ierr);
      ierr = VecCopy(Vn,tmp);CHKERRQ(ierr);
      ierr = VecCopy(Vj,Vn);CHKERRQ(ierr);
      ierr = VecCopy(tmp,Vj);CHKERRQ(ierr);
      
      //ierr = MatDenseRestoreColumnVec(S,last,&Vn);CHKERRQ(ierr);
      //ierr = MatDenseRestoreColumnVec(S,j,&Vj);CHKERRQ(ierr);
      
      ierr = _MatDenseResetColumnVec(S,j,Vj);CHKERRQ(ierr);
      ierr = _MatDenseResetColumnVec(S,last,Vn);CHKERRQ(ierr);
    }
    
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&Vj);CHKERRQ(ierr);
    ierr = VecDestroy(&Vn);CHKERRQ(ierr);
  }

  /* keep the first M*k entries */
  {
    const PetscScalar *_Vko;
    PetscScalar       *_V;
    
    ierr = MatDenseGetArrayRead(S,&_Vko);CHKERRQ(ierr);
    ierr = MatDenseGetArray(V,&_V);CHKERRQ(ierr);
    ierr = PetscMemcpy(_V,_Vko,sizeof(PetscScalar)*M*k);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(V,&_V);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(S,&_Vko);CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin(V,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(V,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode RandEigsMin_MPI_2(KSP ksp,PetscInt k,PetscInt o,PetscInt power_its,PetscRandom prand,Vec *_eigs,Mat *_V)
{
  PetscErrorCode ierr;
  Vec            eigs;
  Mat            A,S,Q,V,R,R_red;
  PetscInt       it,M,m,i;
  PetscMPIInt    commsize,commrank;
  MPI_Comm       comm;
  
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&commrank);CHKERRQ(ierr);
  
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  if (!*_eigs) {
    ierr = VecCreate(PETSC_COMM_SELF,&eigs);CHKERRQ(ierr);
    ierr = VecSetSizes(eigs,PETSC_DECIDE,k);CHKERRQ(ierr);
    ierr = VecSetUp(eigs);CHKERRQ(ierr);
    *_eigs = eigs;
  } else {
    eigs = *_eigs;
  }
  if (!*_V) {
    ierr = MatCreate(comm,&V);CHKERRQ(ierr);
    ierr = MatSetSizes(V,m,PETSC_DECIDE,M,k);CHKERRQ(ierr);
    ierr = MatSetType(V,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(V);CHKERRQ(ierr);
    *_V = V;
  } else {
    V = *_V;
  }
  
  ierr = MatCreate(comm,&S);CHKERRQ(ierr);
  ierr = MatSetSizes(S,m,PETSC_DECIDE,M,(k+o));CHKERRQ(ierr);
  ierr = MatSetType(S,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  
  ierr = MatSetRandom(S,prand);CHKERRQ(ierr);
  //ierr = MatSetRandomConsistent(S);CHKERRQ(ierr);
  //MatView(S,PETSC_VIEWER_STDOUT_WORLD);
  
  //ierr = MatMatMult(A,S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Q);CHKERRQ(ierr); /* aij/baij * dense */
  ierr = MatDenseSolveKSP(ksp,S,MAT_INITIAL_MATRIX,&Q);CHKERRQ(ierr);
  
  if (power_its == 0) {
    ierr = MatOrthogonalizeMatDense(Q);CHKERRQ(ierr);
  } else {
    Mat AQ;
    
    
    //ierr = MatMatMult(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AQ);CHKERRQ(ierr); /* aij/baij * dense */
    ierr = MatDenseSolveKSP(ksp,Q,MAT_INITIAL_MATRIX,&AQ);CHKERRQ(ierr);
    ierr = MatOrthogonalizeMatDense(AQ);CHKERRQ(ierr);
    ierr = MatDenseCopyFast(AQ,Q);CHKERRQ(ierr);
    
    for (it=1; it<power_its; it++) {
      //ierr = MatMatMult(A,Q,MAT_REUSE_MATRIX,PETSC_DEFAULT,&AQ);CHKERRQ(ierr); /* aij/baij * dense */
      ierr = MatDenseSolveKSP(ksp,Q,MAT_REUSE_MATRIX,&AQ);CHKERRQ(ierr);
      ierr = MatOrthogonalizeMatDense(AQ);CHKERRQ(ierr);
      ierr = MatDenseCopyFast(AQ,Q);CHKERRQ(ierr);
    }
    
    ierr = MatDestroy(&AQ);CHKERRQ(ierr);
  }
  
  //ierr = MatPtAP(A,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R);
  {
    Mat AQ;
    
    ierr = MatDenseSolveKSP(ksp,Q,MAT_INITIAL_MATRIX,&AQ);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(Q,AQ,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R);CHKERRQ(ierr);
    
    ierr = MatDestroy(&AQ);CHKERRQ(ierr);
  }
  
  ierr = MatCreateRedundantMatrix(R,commsize,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&R_red);CHKERRQ(ierr);
  
  //(d, V) = eigh(R)
  //ierr = lapackeigs(R_red,eigs);CHKERRQ(ierr);
  {
    Vec               eigs_ko;
    PetscScalar       *_eko,*_e;
    
    ierr = VecCreate(PETSC_COMM_SELF,&eigs_ko);CHKERRQ(ierr);
    ierr = VecSetSizes(eigs_ko,PETSC_DECIDE,k+o);CHKERRQ(ierr);
    ierr = VecSetUp(eigs_ko);CHKERRQ(ierr);
    
    ierr = lapackeigs(R_red,eigs_ko);CHKERRQ(ierr);
    
    /* invert stage of shift-and-invert */
    ierr = VecReciprocal(eigs_ko);CHKERRQ(ierr);
    
    ierr = VecGetArray(eigs_ko,&_eko);CHKERRQ(ierr);
    ierr = VecGetArray(eigs,&_e);CHKERRQ(ierr);
    
    /* reverse the order of the full eigenvector */
    {
      PetscInt    first,last,center;
      PetscScalar tmp;
      
      center = (k+o)/2;
      
      for (i=0; i<center; i++) {
        first = i;
        last = (k+o) - 1 - i;
        
        tmp = _eko[last];
        _eko[last] = _eko[first];
        _eko[first] = tmp;
      }
    }
    
    /* keep the first k terms */
    for (i=0; i<k; i++) {
      _e[i] = _eko[i];
    }
    ierr = VecRestoreArray(eigs,&_e);CHKERRQ(ierr);
    ierr = VecRestoreArray(eigs_ko,&_eko);CHKERRQ(ierr);
    ierr = VecDestroy(&eigs_ko);CHKERRQ(ierr);
  }

  
  
  //V = np.matmul(Q, V)
  //ierr = MatMatMult(Q,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&V);CHKERRQ(ierr); /* dense * seqdesnse */
  
  {
    Mat Q_red,V_red;
    PetscScalar *_array_Q,*_array_V;
    
    ierr = MatDenseGetArray(Q,&_array_Q);CHKERRQ(ierr);
    ierr = MatDenseGetArray(S,&_array_V);CHKERRQ(ierr);
    
    ierr = MatCreate(PETSC_COMM_SELF,&Q_red);CHKERRQ(ierr);
    ierr = MatSetSizes(Q_red,m,(k+o),m,(k+o));CHKERRQ(ierr);
    ierr = MatSetType(Q_red,MATDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(Q_red,_array_Q);CHKERRQ(ierr);
    ierr = MatSetUp(Q_red);CHKERRQ(ierr);
    
    ierr = MatCreate(PETSC_COMM_SELF,&V_red);CHKERRQ(ierr);
    ierr = MatSetSizes(V_red,m,(k+o),m,(k+o));CHKERRQ(ierr);
    ierr = MatSetType(V_red,MATDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(V_red,_array_V);CHKERRQ(ierr);
    ierr = MatSetUp(V_red);CHKERRQ(ierr);
    
    ierr = MatMatMult(Q_red,R_red,MAT_REUSE_MATRIX,PETSC_DEFAULT,&V_red);CHKERRQ(ierr);
    
    ierr = MatDestroy(&V_red);CHKERRQ(ierr);
    ierr = MatDestroy(&Q_red);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(Q,&_array_Q);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(S,&_array_V);CHKERRQ(ierr);
  }
  
  /* reverse order of eigenvectors */
  {
    PetscInt j,first,last,center;
    Vec      tmp,Vj,Vn;
    
    center = (k+o)/2;
    
    ierr = MatCreateVecs(S,NULL,&tmp);CHKERRQ(ierr);
    ierr = _MatDenseCreateEmptyColumnVec(S,&Vj);CHKERRQ(ierr);
    ierr = _MatDenseCreateEmptyColumnVec(S,&Vn);CHKERRQ(ierr);
    
    for (j=0; j<center; j++) {
      first = j;
      last = (k+o) - 1 - j;
      
      ierr = _MatDenseSetColumnVec(S,j,Vj);CHKERRQ(ierr);
      ierr = _MatDenseSetColumnVec(S,last,Vn);CHKERRQ(ierr);
      
      //ierr = MatDenseGetColumnVec(S,j,&Vj);CHKERRQ(ierr);
      //ierr = MatDenseGetColumnVec(S,last,&Vn);CHKERRQ(ierr);
      ierr = VecCopy(Vn,tmp);CHKERRQ(ierr);
      ierr = VecCopy(Vj,Vn);CHKERRQ(ierr);
      ierr = VecCopy(tmp,Vj);CHKERRQ(ierr);
      
      //ierr = MatDenseRestoreColumnVec(S,last,&Vn);CHKERRQ(ierr);
      //ierr = MatDenseRestoreColumnVec(S,j,&Vj);CHKERRQ(ierr);
      
      ierr = _MatDenseResetColumnVec(S,j,Vj);CHKERRQ(ierr);
      ierr = _MatDenseResetColumnVec(S,last,Vn);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&Vj);CHKERRQ(ierr);
    ierr = VecDestroy(&Vn);CHKERRQ(ierr);
  }
  
  /* keep the first m*k entries */
  {
    const PetscScalar *_Vko;
    PetscScalar       *_V;
    
    ierr = MatDenseGetArrayRead(S,&_Vko);CHKERRQ(ierr);
    ierr = MatDenseGetArray(V,&_V);CHKERRQ(ierr);
    ierr = PetscMemcpy(_V,_Vko,sizeof(PetscScalar)*m*k);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(V,&_V);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(S,&_Vko);CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin(V,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(V,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  
  ierr = MatDestroy(&R_red);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

