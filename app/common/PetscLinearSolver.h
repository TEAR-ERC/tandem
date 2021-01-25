#ifndef PETSCSOLVER_20200910_H
#define PETSCSOLVER_20200910_H

#include "PetscUtil.h"
#include "common/PetscMatrix.h"
#include "common/PetscVector.h"

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
        A_ = std::make_unique<PetscMatrix>(dgop.block_size(), topo.numLocalElements(),
                                           topo.numLocalNeighbours(), topo.numGhostNeighbours(),
                                           topo.comm());
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
    void solve() { CHKERRTHROW(KSPSolve(ksp_, b_->vec(), x_->vec())); }

    auto& x() { return *x_; }
    auto const& x() const { return *x_; }

    KSP ksp() { return ksp_; }

    void dump() const {
        PetscViewer viewer;
        CHKERRTHROW(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
        CHKERRTHROW(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
        CHKERRTHROW(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));

        CHKERRTHROW(PetscViewerFileSetName(viewer, "A.bin"));
        CHKERRTHROW(MatView(A_->mat(), viewer));

        CHKERRTHROW(PetscViewerFileSetName(viewer, "b.bin"));
        CHKERRTHROW(VecView(b_->vec(), viewer));

        CHKERRTHROW(PetscViewerDestroy(&viewer));
    }

private:
    std::unique_ptr<PetscMatrix> A_;
    std::unique_ptr<PetscVector> b_;
    std::unique_ptr<PetscVector> x_;
    KSP ksp_ = nullptr;
};

} // namespace tndm

#endif // PETSCSOLVER_20200910_H
