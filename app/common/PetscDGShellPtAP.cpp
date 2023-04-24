#include "PetscDGShellPtAP.h"
#include "PetscDGMatrix.h"
#include "common/PetscUtil.h"
#include "common/PetscVector.h"

namespace tndm {

PetscDGShellPtAP::PetscDGShellPtAP(AbstractDGOperator<DomainDimension>& dgop, Mat P, std::size_t to_blocksize) {
    const auto blockSize = dgop.block_size();
    const auto localSize = blockSize * dgop.num_local_elements();
    const auto comm = dgop.topo().comm();
    PetscInt m,n;

    CHKERRTHROW(MatGetLocalSize(P,&m,&n));
    //printf("to %zu\n",to_blocksize);

    CHKERRTHROW(MatCreate(comm, &A_));
    CHKERRTHROW(MatSetSizes(A_, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRTHROW(MatSetBlockSize(A_, to_blocksize));
    CHKERRTHROW(MatSetType(A_, MATSHELL));
    CHKERRTHROW(MatSetUp(A_));

    //CHKERRTHROW(MatShellSetContext(A_, static_cast<void*>(&dgop)));
    //printf("shell->ctx %p\n",(void*)this);
    CHKERRTHROW(MatShellSetContext(A_, (void*)this));
    CHKERRTHROW(MatShellSetOperation(A_, MATOP_MULT, (void (*)(void))apply));
    CHKERRTHROW(MatShellSetOperation(A_, MATOP_DESTROY, (void (*)(void))destroy));

    dgop_ = &dgop;
    blocksize_ = blockSize;
    to_blocksize_ = to_blocksize;
    P_ = P;

    CHKERRTHROW(PetscObjectReference((PetscObject)P));
    CHKERRTHROW(MatCreateVecs(P, NULL, &xf_));
    CHKERRTHROW(VecDuplicate(xf_, &yf_));
}

PetscDGShellPtAP::~PetscDGShellPtAP() {
    //printf("class destroy\n");
    MatDestroy(&A_);
    VecDestroy(&xf_);
    VecDestroy(&yf_);
    MatDestroy(&P_);
}

Mat PetscDGShellPtAP::create_explicit(DGOperatorTopo const& topo, DGOpSparsityType stype) {

  auto asmA = std::make_unique<PetscDGMatrix>(to_blocksize_, topo, stype);
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

PetscErrorCode PetscDGShellPtAP::destroy(Mat A) {
    PetscDGShellPtAP *ctx;

    //printf("mat destroy\n");
    MatShellGetContext(A, (void**)&ctx);
    VecDestroy(&ctx->xf_);
    VecDestroy(&ctx->yf_);
    MatDestroy(&ctx->P_);
    return 0;
}

PetscErrorCode PetscDGShellPtAP::apply(Mat A, Vec x, Vec y) {
    //AbstractDGOperator<DomainDimension>* dgop;
    //MatShellGetContext(A, &dgop);
    PetscDGShellPtAP *ctx;
    MatShellGetContext(A, (void**)&ctx);
    AbstractDGOperator<DomainDimension>* dgop = ctx->dgop_;
    //

    MatMult(ctx->P_, x, ctx->xf_);
    const auto xv = PetscVectorView(ctx->xf_);
    auto yv = PetscVectorView(ctx->yf_);
    dgop->apply(xv, yv);
    MatMultTranspose(ctx->P_, ctx->yf_, y);
    return 0;
}

PetscErrorCode PetscDGShellPtAP::apply_f(Mat A, Vec x, Vec y, void *ctx)
{
  CHKERRTHROW(apply(A, x, y));
  return 0;
}


} // namespace tndm
