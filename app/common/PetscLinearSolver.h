#ifndef PETSCSOLVER_20200910_H
#define PETSCSOLVER_20200910_H

#include "common/MGConfig.h"
#include "common/PetscDGMatrix.h"
#include "common/PetscDGShell.h"
#include "common/PetscInterplMatrix.h"
#include "common/PetscUtil.h"
#include "common/PetscVector.h"
#include "config.h"

#include "form/InterpolationOperator.h"

#include <mpi.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <cstddef>
#include <experimental/type_traits>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

class PetscLinearSolver {
public:
    template <typename DGOp>
    PetscLinearSolver(DGOp& dgop, bool matrix_free = false,
                      MGConfig const& mg_config = MGConfig()) {
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
            setup_mg(dgop, pc, mg_config);
            break;
        default:
            break;
        };
    }
    ~PetscLinearSolver() {
        for (auto&& A : mat_cleanup) {
            MatDestroy(&A);
        }
        KSPDestroy(&ksp_);
    }

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
    template <class LocalOperator>
    using make_interpolation_op_t = decltype(&LocalOperator::make_interpolation_op);

    template <typename DGOp> void setup_mg(DGOp& dgop, PC pc, MGConfig const& mg_config) {
        if constexpr (std::experimental::is_detected_v<make_interpolation_op_t,
                                                       typename DGOp::local_operator_t>) {
            auto i_op = InterpolationOperator(dgop.topo().numLocalElements(),
                                              dgop.lop().make_interpolation_op());

            auto level_degree = mg_config.levels(i_op.max_degree());
            unsigned nlevels = level_degree.size();

            int rank;
            MPI_Comm_rank(dgop.topo().comm(), &rank);
            if (rank == 0) {
                std::cout << "Multigrid P-levels: ";
                for (auto level : level_degree) {
                    std::cout << level << " ";
                }
                std::cout << std::endl;
            }

            CHKERRTHROW(PCMGSetLevels(pc, nlevels, nullptr));
            CHKERRTHROW(PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE));
            KSP smooth;
            CHKERRTHROW(PCMGGetSmoother(pc, nlevels - 1, &smooth));
            if (A_) {
                CHKERRTHROW(KSPSetOperators(smooth, A_->mat(), P_->mat()));
            } else {
                CHKERRTHROW(KSPSetOperators(smooth, P_->mat(), P_->mat()));
            }
            Mat A_lp1 = P_->mat();
            for (int l = nlevels - 2; l >= 0; --l) {
                unsigned to_degree = level_degree[l + 1];
                unsigned from_degree = level_degree[l];
                auto I = PetscInterplMatrix(i_op.block_size(to_degree),
                                            i_op.block_size(from_degree), dgop.topo());
                i_op.assemble(to_degree, from_degree, I);
                CHKERRTHROW(PCMGSetInterpolation(pc, l + 1, I.mat()));

                Mat A_l;
                CHKERRTHROW(MatPtAP(A_lp1, I.mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A_l));
                CHKERRTHROW(PCMGGetSmoother(pc, l, &smooth));
                CHKERRTHROW(KSPSetOperators(smooth, A_l, A_l));
                mat_cleanup.push_back(A_l);
                A_lp1 = A_l;
            }
        } else {
            throw std::logic_error("interpolation not set up for selected local operator");
        }
    }
    void warmup_ksp(KSP ksp);
    void warmup_sub_pcs(PC pc);
    void warmup_composite(PC pc);
    void warmup_mg(PC pc);

    std::unique_ptr<PetscDGShell> A_;
    std::unique_ptr<PetscDGMatrix> P_;
    std::unique_ptr<PetscVector> b_;
    std::unique_ptr<PetscVector> x_;
    KSP ksp_ = nullptr;

    std::vector<Mat> mat_cleanup;
};

} // namespace tndm

#endif // PETSCSOLVER_20200910_H
