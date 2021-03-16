#ifndef PETSCSOLVER_20200910_H
#define PETSCSOLVER_20200910_H

#include "PetscUtil.h"
#include "common/PetscDGMatrix.h"
#include "common/PetscDGShell.h"
#include "common/PetscInterplMatrix.h"
#include "common/PetscVector.h"

#include <petscksp.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

class PetscLinearSolver {
public:
    template <typename DGOp> PetscLinearSolver(DGOp& dgop, bool matrix_free = false) {
        auto const& topo = dgop.topo();
        if (matrix_free) {
            A_ = std::make_unique<PetscDGShell>(dgop);
        }

        P_ = std::make_unique<PetscDGMatrix>(dgop.block_size(), topo);
        dgop.assemble(*P_);

        b_ = std::make_unique<PetscVector>(dgop.block_size(), topo.numLocalElements(), topo.comm());
        x_ = std::make_unique<PetscVector>(*b_);
        dgop.rhs(*b_);

        CHKERRTHROW(KSPCreate(topo.comm(), &ksp_));
        CHKERRTHROW(KSPSetType(ksp_, KSPCG));
        if (matrix_free) {
            CHKERRTHROW(KSPSetOperators(ksp_, A_->mat(), P_->mat()));
        } else {
            CHKERRTHROW(KSPSetOperators(ksp_, P_->mat(), P_->mat()));
        }
        CHKERRTHROW(KSPSetTolerances(ksp_, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
        CHKERRTHROW(KSPSetFromOptions(ksp_));

        PC pc;
        CHKERRTHROW(KSPGetPC(ksp_, &pc));
        PCType type;
        PCGetType(pc, &type);
        switch (fnv1a(type)) {
        case HASH_DEF(PCMG):
            setup_mg(dgop, pc);
            break;
        default:
            break;
        };
    }
    ~PetscLinearSolver() { KSPDestroy(&ksp_); }

    template <typename DGOp> void update_rhs(DGOp& dgop) {
        b_->set_zero();
        dgop.rhs(*b_);
    }
    void warmup();
    void solve() { CHKERRTHROW(KSPSolve(ksp_, b_->vec(), x_->vec())); }
    bool is_converged() const {
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp_, &reason);
        return reason > 0;
    }

    auto& x() { return *x_; }
    auto const& x() const { return *x_; }

    KSP ksp() { return ksp_; }

    void dump() const;

private:
    template <typename DGOp> void setup_mg(DGOp& dgop, PC pc) {
        auto const& topo = dgop.topo();
        CHKERRTHROW(PCMGSetLevels(pc, dgop.num_levels() + 1, nullptr));
        CHKERRTHROW(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
        for (unsigned l = 0; l < dgop.num_levels(); ++l) {
            auto Interpl =
                PetscInterplMatrix(dgop.block_size_level(l + 1), dgop.block_size_level(l), topo);
            dgop.assemble_interpolate(l, Interpl);

            CHKERRTHROW(PCMGSetInterpolation(pc, l + 1, Interpl.mat()));
        }
    }
    void warmup_sub_pcs(PC pc);
    void warmup_composite(PC pc);

    std::unique_ptr<PetscDGShell> A_;
    std::unique_ptr<PetscDGMatrix> P_;
    std::unique_ptr<PetscVector> b_;
    std::unique_ptr<PetscVector> x_;
    KSP ksp_ = nullptr;
};

} // namespace tndm

#endif // PETSCSOLVER_20200910_H
