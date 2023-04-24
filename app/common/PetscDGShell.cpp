#include "PetscDGShell.h"
#include "PetscDGMatrix.h"
#include "common/PetscUtil.h"
#include "common/PetscVector.h"

namespace tndm {

PetscDGShell::PetscDGShell(AbstractDGOperator<DomainDimension>& dgop) {
    const auto blockSize = dgop.block_size();
    const auto localSize = blockSize * dgop.num_local_elements();
    const auto comm = dgop.topo().comm();
    CHKERRTHROW(MatCreate(comm, &A_));
    CHKERRTHROW(MatSetSizes(A_, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRTHROW(MatSetBlockSize(A_, blockSize));
    CHKERRTHROW(MatSetType(A_, MATSHELL));
    CHKERRTHROW(MatSetUp(A_));

    CHKERRTHROW(MatShellSetContext(A_, static_cast<void*>(&dgop)));
    CHKERRTHROW(MatShellSetOperation(A_, MATOP_MULT, (void (*)(void))apply));
    blocksize_ = blockSize;
}

PetscDGShell::~PetscDGShell() { MatDestroy(&A_); }

Mat PetscDGShell::create_explicit(DGOperatorTopo const& topo, DGOpSparsityType stype) {

  auto asmA = std::make_unique<PetscDGMatrix>(blocksize_, topo, stype);
  Mat asmA_ = asmA->mat();
  Vec x;
  MatColoring mc;
  ISColoring isc;
  MatFDColoring color;
  void *ctx = NULL;

  CHKERRTHROW(PetscObjectReference((PetscObject)asmA_));
  CHKERRTHROW(MatShellGetContext(A_, (void**)&ctx));
  CHKERRTHROW(MatColoringCreate(asmA_, &mc));
  CHKERRTHROW(MatColoringSetDistance(mc, 2));
  CHKERRTHROW(MatColoringSetType(mc, MATCOLORINGJP));
  CHKERRTHROW(MatColoringSetFromOptions(mc));
  CHKERRTHROW(MatColoringApply(mc, &isc));
  CHKERRTHROW(MatColoringDestroy(&mc));

  CHKERRTHROW(MatFDColoringCreate(asmA_, isc, &color));

  CHKERRTHROW(MatFDColoringSetFunction(color, (PetscErrorCode(*)(void))apply_f, NULL));

  CHKERRTHROW(MatFDColoringSetFromOptions(color));
  CHKERRTHROW(MatFDColoringSetUp(asmA_, isc, color));
  CHKERRTHROW(ISColoringDestroy(&isc));

  CHKERRTHROW(MatCreateVecs(asmA_, &x, NULL));
  CHKERRTHROW(MatFDColoringApply(asmA_, color, x, (void*)A_));

  CHKERRTHROW(MatFDColoringDestroy(&color));
  CHKERRTHROW(VecDestroy(&x));

  return asmA_;
}

PetscErrorCode PetscDGShell::apply(Mat A, Vec x, Vec y) {
    AbstractDGOperator<DomainDimension>* dgop;
    MatShellGetContext(A, &dgop);
    const auto xv = PetscVectorView(x);
    auto yv = PetscVectorView(y);
    dgop->apply(xv, yv);
    return 0;
}

PetscErrorCode PetscDGShell::apply_f(Mat A, Vec x, Vec y, void *ctx)
{
  CHKERRTHROW(apply(A, x, y));
  return 0;
}


} // namespace tndm
