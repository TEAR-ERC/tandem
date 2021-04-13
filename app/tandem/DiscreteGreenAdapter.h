#ifndef DISCRETEGREENADAPTER_20210412_H
#define DISCRETEGREENADAPTER_20210412_H

#include "common/PetscUtil.h"
#include "common/PetscVector.h"

#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"
#include "util/LinearAllocator.h"
#include "util/Scratch.h"

#include <petscmat.h>

#include <iostream>
#include <memory>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <utility>

namespace tndm {

template <typename Adapter> class DiscreteGreenAdapter {
public:
    using time_functional_t = typename Adapter::time_functional_t;

    DiscreteGreenAdapter(std::unique_ptr<Adapter> adapter, std::size_t slip_block_size)
        : adapter_(std::move(adapter)), slip_block_size_(slip_block_size) {}
    ~DiscreteGreenAdapter() { MatDestroy(&T_); }

    void set_boundary(time_functional_t fun) {
        adapter_->set_boundary(std::move(fun));
        compute_boundary_traction();
    }

    std::size_t scratch_mem_size() const { return adapter_->scratch_mem_size(); }
    auto const& topo() const { return adapter_->topo(); }
    auto const& faultMap() const { return adapter_->faultMap(); }

    void begin_preparation(std::size_t numFaultFaces) {
        adapter_->begin_preparation(numFaultFaces);
        numFaultFaces_ = numFaultFaces;
    }
    void prepare(std::size_t faultNo, LinearAllocator<double>& scratch) {
        adapter_->prepare(faultNo, scratch);
    }
    void end_preparation() {
        adapter_->end_preparation();
        compute_discrete_greens_function();
    }

    template <typename BlockVector> void solve(double time, BlockVector& state) {
        auto in_handle = state.begin_access();
        for (std::size_t faultNo = 0; faultNo < numFaultFaces_; ++faultNo) {
            auto state_block = state.get_block(in_handle, faultNo);
            S_->insert_block(faultNo, state_block);
        }
        S_->begin_assembly();
        S_->end_assembly();
        state.end_access(in_handle);

        CHKERRTHROW(MatMult(T_, S_->vec(), t_->vec()));
        CHKERRTHROW(VecAXPY(t_->vec(), time, t_boundary_->vec()));
    }

    template <typename BlockVector>
    void full_solve(double time, BlockVector& state, bool reuse_last_solve) {
        adapter_->solve(time, state);
    }

    TensorBase<Matrix<double>> traction_info() const { return adapter_->traction_info(); }
    template <class Func> void begin_traction(Func func) { handle_ = t_->begin_access_readonly(); }
    void traction(std::size_t faultNo, Matrix<double>& traction,
                  LinearAllocator<double>& blabla) const {
        auto block = t_->get_block(handle_, faultNo);
        assert(block.size() == traction.size());

        for (int i = 0; i < block.size(); ++i) {
            traction.data()[i] = block.data()[i];
        }
    }
    void end_traction() { t_->end_access_readonly(handle_); }

    auto displacement() const { return adapter_->displacement(); }
    std::size_t numLocalElements() const { return adapter_->numLocalElements(); }
    std::size_t number_of_local_dofs() const { return adapter_->number_of_local_dofs(); }

private:
    void compute_discrete_greens_function();
    void compute_boundary_traction();

    std::unique_ptr<Adapter> adapter_;
    std::size_t slip_block_size_;

    std::size_t numFaultFaces_ = 0;
    Mat T_ = nullptr;
    std::unique_ptr<PetscVector> S_;
    std::unique_ptr<PetscVector> t_boundary_;
    std::unique_ptr<PetscVector> t_;
    PetscVector::const_handle handle_;
};

template <typename Adapter> void DiscreteGreenAdapter<Adapter>::compute_discrete_greens_function() {
    auto scratch = Scratch<double>(adapter_->scratch_mem_size(), ALIGNMENT);
    auto traction = Managed<Matrix<double>>(adapter_->traction_info());

    PetscInt m_bs = traction.size();
    PetscInt n_bs = 1;
    PetscInt m = numFaultFaces_ * m_bs;
    PetscInt n = numFaultFaces_ * slip_block_size_ * n_bs;

    MPI_Comm comm = topo().comm();

    int rank;
    MPI_Comm_rank(comm, &rank);

    PetscInt mb_offset = 0;
    PetscInt nb_offset = 0;
    MPI_Scan(&numFaultFaces_, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
    mb_offset -= numFaultFaces_;
    MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
    nb_offset -= n;

    CHKERRTHROW(MatCreateDense(comm, m, n, PETSC_DECIDE, PETSC_DECIDE, nullptr, &T_));
    CHKERRTHROW(MatSetBlockSizes(T_, m_bs, n_bs));

    S_ = std::make_unique<PetscVector>(slip_block_size_, numFaultFaces_, comm);
    t_ = std::make_unique<PetscVector>(m_bs, numFaultFaces_, comm);
    t_boundary_ = std::make_unique<PetscVector>(m_bs, numFaultFaces_, comm);

    PetscInt N;
    CHKERRTHROW(VecGetSize(S_->vec(), &N));

    for (PetscInt i = 0; i < N; ++i) {
        if (rank == 0) {
            std::cout << "Computing Green's function " << (i + 1) << "/" << N << std::endl;
        }
        CHKERRTHROW(VecZeroEntries(S_->vec()));
        if (i >= nb_offset && i < nb_offset + m) {
            PetscScalar one = 1.0;
            CHKERRTHROW(VecSetValue(S_->vec(), i, one, INSERT_VALUES));
        }
        S_->begin_assembly();
        S_->end_assembly();

        adapter_->solve(0.0, *S_);

        auto S_handle = S_->begin_access_readonly();
        adapter_->begin_traction(
            [this, &S_handle](std::size_t faultNo) { return S_->get_block(S_handle, faultNo); });
        for (std::size_t faultNo = 0; faultNo < numFaultFaces_; ++faultNo) {
            scratch.reset();
            adapter_->traction(faultNo, traction, scratch);
            PetscInt g_m = mb_offset + faultNo;
            PetscInt g_n = i;
            CHKERRTHROW(MatSetValuesBlocked(T_, 1, &g_m, 1, &g_n, traction.data(), INSERT_VALUES));
        }
        adapter_->end_traction();
        S_->end_access_readonly(S_handle);
    }

    CHKERRTHROW(MatAssemblyBegin(T_, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd(T_, MAT_FINAL_ASSEMBLY));
}

template <typename Adapter> void DiscreteGreenAdapter<Adapter>::compute_boundary_traction() {
    MPI_Comm comm = topo().comm();
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::cout << "Computing boundary Green's function" << std::endl;
    }

    auto scratch = Scratch<double>(adapter_->scratch_mem_size(), ALIGNMENT);
    auto traction = Managed<Matrix<double>>(adapter_->traction_info());

    CHKERRTHROW(VecZeroEntries(S_->vec()));
    adapter_->solve(1.0, *S_);

    auto S_handle = S_->begin_access_readonly();
    adapter_->begin_traction(
        [this, &S_handle](std::size_t faultNo) { return S_->get_block(S_handle, faultNo); });
    for (std::size_t faultNo = 0; faultNo < numFaultFaces_; ++faultNo) {
        scratch.reset();
        adapter_->traction(faultNo, traction, scratch);
        PetscInt l_m = faultNo;
        CHKERRTHROW(
            VecSetValuesBlockedLocal(t_boundary_->vec(), 1, &l_m, traction.data(), INSERT_VALUES));
    }
    t_boundary_->begin_assembly();
    t_boundary_->end_assembly();
    adapter_->end_traction();
    S_->end_access_readonly(S_handle);
}

} // namespace tndm

#endif // DISCRETEGREENADAPTER_20210412_H
