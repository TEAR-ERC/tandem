#ifndef PETSCSOLVER_20200910_H
#define PETSCSOLVER_20200910_H

#include "PetscUtil.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "tensor/Tensor.h"

#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <cassert>
#include <cstddef>

namespace tndm {

class PetscSolver {
public:
    ~PetscSolver() {
        if (A_ != nullptr) {
            MatDestroy(&A_);
        }
        if (b_ != nullptr) {
            VecDestroy(&b_);
        }
        if (x_ != nullptr) {
            VecDestroy(&x_);
        }
        if (ksp_ != nullptr) {
            KSPDestroy(&ksp_);
        }
    }

    void create_mat(std::size_t blockSize, std::size_t numLocalElems, unsigned const* numLocal,
                    unsigned const* numGhost, bool reuse, MPI_Comm comm);
    void create_vec(std::size_t blockSize, std::size_t numLocalElems, bool reuse, MPI_Comm comm);

    void begin_assembly() {}
    void add_block(std::size_t ib, std::size_t jb, Matrix<double> const& A) {
        assert(A_ != nullptr);
        PetscInt pib = ib;
        PetscInt pjb = jb;
        MatSetValuesBlocked(A_, 1, &pib, 1, &pjb, A.data(), ADD_VALUES);
    }
    void end_assembly() {
        CHKERRTHROW(MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY));
        CHKERRTHROW(MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY));
    }

    void begin_rhs() {}
    void add_rhs(std::size_t ib, Vector<double> const& B) {
        assert(b_ != nullptr);
        PetscInt pib = ib;
        VecSetValuesBlocked(b_, 1, &pib, B.data(), ADD_VALUES);
    }
    void end_rhs() {
        CHKERRTHROW(VecAssemblyBegin(b_));
        CHKERRTHROW(VecAssemblyEnd(b_));
    }

    void setup_solve();
    void solve() {
        if (!ksp_) {
            setup_solve();
        }
        if (x_ == nullptr) {
            CHKERRTHROW(VecDuplicate(b_, &x_));
        }

        CHKERRTHROW(KSPSolve(ksp_, b_, x_));
    }

    // TODO: Shouldn't really be here -> move
    template <std::size_t D> FiniteElementFunction<D> solution(RefElement<D> const& space) const {
        assert(blockSize_ != 0 && blockSize_ % space.numBasisFunctions() == 0);
        auto numQuantities = blockSize_ / space.numBasisFunctions();
        PetscScalar const* values;
        VecGetArrayRead(x_, &values);
        auto numeric =
            FiniteElementFunction<D>(space.clone(), values, numQuantities, numLocalElems_);
        VecRestoreArrayRead(x_, &values);
        return numeric;
    }

    KSP ksp() { return ksp_; }

private:
    Mat A_ = nullptr;
    Vec b_ = nullptr;
    Vec x_ = nullptr;
    KSP ksp_ = nullptr;
    std::size_t numLocalElems_ = 0;
    std::size_t blockSize_ = 0;
};

} // namespace tndm

#endif // PETSCSOLVER_20200910_H
