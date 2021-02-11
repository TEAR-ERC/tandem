#ifndef PETSCSOLVER_20200910_H
#define PETSCSOLVER_20200910_H

#include "PetscUtil.h"
#include "common/PetscDGMatrix.h"
#include "common/PetscVector.h"

#include <petscksp.h>
#include <petscpc.h>
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
        A_ = std::make_unique<PetscDGMatrix>(dgop.block_size(), topo);
        b_ = std::make_unique<PetscVector>(dgop.block_size(), topo.numLocalElements(), topo.comm());
        x_ = std::make_unique<PetscVector>(*b_);
        dgop.assemble(*A_);
        dgop.rhs(*b_);

        CHKERRTHROW(KSPCreate(topo.comm(), &ksp_));
        CHKERRTHROW(KSPSetType(ksp_, KSPCG));
        CHKERRTHROW(KSPSetOperators(ksp_, A_->mat(), A_->mat()));
        CHKERRTHROW(KSPSetTolerances(ksp_, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
        CHKERRTHROW(KSPSetFromOptions(ksp_));
    }
    ~PetscLinearSolver() { KSPDestroy(&ksp_); }

    template <typename DGOp> void update_rhs(DGOp& dgop) {
        b_->set_zero();
        dgop.rhs(*b_);
    }
    void warmup();
    void solve() { CHKERRTHROW(KSPSolve(ksp_, b_->vec(), x_->vec())); }

    auto& x() { return *x_; }
    auto const& x() const { return *x_; }

    KSP ksp() { return ksp_; }

    void dump() const;

private:
    void warmup_composite(PC pc);

    std::unique_ptr<PetscDGMatrix> A_;
    std::unique_ptr<PetscVector> b_;
    std::unique_ptr<PetscVector> x_;
    KSP ksp_ = nullptr;
};

} // namespace tndm

#endif // PETSCSOLVER_20200910_H
