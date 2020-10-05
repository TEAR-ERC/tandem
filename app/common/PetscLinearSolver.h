#ifndef PETSCSOLVER_20200910_H
#define PETSCSOLVER_20200910_H

#include "PetscUtil.h"
#include "common/PetscBlockMatrix.h"
#include "common/PetscBlockVector.h"

#include <petscksp.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

class PetscLinearSolver {
public:
    template <typename DGOp> PetscLinearSolver(DGOp& dgop) {
        auto const& topo = dgop.topo();
        A_ = std::make_unique<PetscBlockMatrix>(dgop.block_size(), topo.numLocalElements(),
                                                topo.numLocalNeighbours(),
                                                topo.numGhostNeighbours(), topo.comm());
        b_ = std::make_unique<PetscBlockVector>(dgop.block_size(), topo.numLocalElements(),
                                                topo.comm());
        x_ = std::make_unique<PetscBlockVector>(*b_);
        dgop.assemble(*A_);
        dgop.rhs(*b_);

        CHKERRTHROW(KSPCreate(PETSC_COMM_WORLD, &ksp_));
        CHKERRTHROW(KSPSetType(ksp_, KSPCG));
        CHKERRTHROW(KSPSetOperators(ksp_, A_->mat(), A_->mat()));
        CHKERRTHROW(KSPSetTolerances(ksp_, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
        CHKERRTHROW(KSPSetFromOptions(ksp_));
    }
    ~PetscLinearSolver() { KSPDestroy(&ksp_); }

    void solve() { CHKERRTHROW(KSPSolve(ksp_, b_->vec(), x_->vec())); }

    auto& x() { return *x_; }
    auto const& x() const { return *x_; }

    KSP ksp() { return ksp_; }

private:
    std::unique_ptr<PetscBlockMatrix> A_;
    std::unique_ptr<PetscBlockVector> b_;
    std::unique_ptr<PetscBlockVector> x_;
    KSP ksp_ = nullptr;
};

} // namespace tndm

#endif // PETSCSOLVER_20200910_H
