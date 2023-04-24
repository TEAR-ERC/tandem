#ifndef PETSCSOLVER_20200910_H
#define PETSCSOLVER_20200910_H

#include "common/MGConfig.h"
#include "common/PetscDGMatrix.h"
#include "common/PetscDGShell.h"
#include "common/PetscDGShellPtAP.h"
#include "common/PetscInterplMatrix.h"
#include "common/PetscUtil.h"
#include "common/PetscVector.h"
#include "config.h"

#include "form/AbstractDGOperator.h"
#include "form/InterpolationOperator.h"

#include <mpi.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

class PetscLinearSolver {
public:
    PetscLinearSolver(AbstractDGOperator<DomainDimension>& dgop, bool matrix_free = false,
                      MGConfig const& mg_config = MGConfig());
    ~PetscLinearSolver();

    inline void update_rhs(AbstractDGOperator<DomainDimension>& dgop) {
        b_->set_zero();
        dgop.rhs(*b_);
    }
    void warmup();
    inline void solve() { CHKERRTHROW(KSPSolve(ksp_, b_->vec(), x_->vec())); }
    inline bool is_converged() const {
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp_, &reason);
        return reason > 0;
    }

    inline auto& x() { return *x_; }
    inline auto const& x() const { return *x_; }

    inline KSP ksp() { return ksp_; }

    void dump() const;

private:
    void setup_mg_default_asm(AbstractDGOperator<DomainDimension>& dgop, PC pc, MGConfig const& mg_config);
    void setup_mg_default_mf(AbstractDGOperator<DomainDimension>& dgop, PC pc, MGConfig const& mg_config);
    void setup_mg(AbstractDGOperator<DomainDimension>& dgop, PC pc, MGConfig const& mg_config);

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
