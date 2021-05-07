#ifndef DISCRETEGREENADAPTER_20210412_H
#define DISCRETEGREENADAPTER_20210412_H

#include "common/PetscUtil.h"
#include "common/PetscVector.h"

#include "interface/BlockView.h"
#include "parallel/LocalGhostCompositeView.h"
#include "parallel/Scatter.h"
#include "parallel/SparseBlockVector.h"
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
    }
    void prepare(std::size_t faultNo, LinearAllocator<double>& scratch) {
        adapter_->prepare(faultNo, scratch);
    }
    void end_preparation() {
        adapter_->end_preparation();
        compute_discrete_greens_function();
    }

    void solve(double time, BlockView const& state) {
        for (std::size_t faultNo = 0, num = adapter_->faultMap().local_size(); faultNo < num;
             ++faultNo) {
            S_->insert_block(faultNo, state.get_block(faultNo));
        }
        S_->begin_assembly();
        S_->end_assembly();

        CHKERRTHROW(MatMult(T_, S_->vec(), t_->vec()));
        CHKERRTHROW(VecAXPY(t_->vec(), time, t_boundary_->vec()));
    }

    void full_solve(double time, BlockView const& state, bool reuse_last_solve) {
        adapter_->solve(time, state);
    }

    TensorBase<Matrix<double>> traction_info() const { return adapter_->traction_info(); }
    void begin_traction(BlockView const&) { handle_ = t_->begin_access_readonly(); }
    void traction(std::size_t faultNo, Matrix<double>& traction, LinearAllocator<double>&) const {
        auto block = handle_.subtensor(slice{}, faultNo);
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

    Mat T_ = nullptr;
    std::unique_ptr<PetscVector> S_;
    std::unique_ptr<PetscVector> t_boundary_;
    std::unique_ptr<PetscVector> t_;
    Matrix<const double> handle_;
};

template <typename Adapter> void DiscreteGreenAdapter<Adapter>::compute_discrete_greens_function() {
    auto scratch = Scratch<double>(adapter_->scratch_mem_size(), ALIGNMENT);
    auto traction = Managed<Matrix<double>>(adapter_->traction_info());

    PetscInt numFaultFaces = adapter_->faultMap().local_size();
    PetscInt m_bs = traction.size();
    PetscInt n_bs = 1;
    PetscInt m = numFaultFaces * m_bs;
    PetscInt n = numFaultFaces * slip_block_size_ * n_bs;

    MPI_Comm comm = topo().comm();

    int rank;
    MPI_Comm_rank(comm, &rank);

    PetscInt mb_offset = 0;
    PetscInt nb_offset = 0;
    MPI_Scan(&numFaultFaces, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
    mb_offset -= numFaultFaces;
    MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
    nb_offset -= n;

    CHKERRTHROW(MatCreateDense(comm, m, n, PETSC_DECIDE, PETSC_DECIDE, nullptr, &T_));
    CHKERRTHROW(MatSetBlockSizes(T_, m_bs, n_bs));

    auto scatter = Scatter(adapter_->faultMap().scatter_plan());
    auto ghost = scatter.template recv_prototype<double>(slip_block_size_, ALIGNMENT);
    S_ = std::make_unique<PetscVector>(slip_block_size_, numFaultFaces, comm);

    t_ = std::make_unique<PetscVector>(m_bs, numFaultFaces, comm);
    t_boundary_ = std::make_unique<PetscVector>(m_bs, numFaultFaces, comm);

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

        scatter.begin_scatter(*S_, ghost);
        scatter.wait_scatter();

        auto S_handle = S_->begin_access_readonly();
        auto block_view = LocalGhostCompositeView(S_handle, ghost);
        adapter_->solve(0.0, block_view);

        adapter_->begin_traction(block_view);
        for (std::size_t faultNo = 0; faultNo < numFaultFaces; ++faultNo) {
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

    auto scatter = Scatter(adapter_->faultMap().scatter_plan());
    auto ghost = scatter.template recv_prototype<double>(slip_block_size_, ALIGNMENT);

    CHKERRTHROW(VecZeroEntries(S_->vec()));
    scatter.begin_scatter(*S_, ghost);
    scatter.wait_scatter();

    auto S_handle = S_->begin_access_readonly();
    auto block_view = LocalGhostCompositeView(S_handle, ghost);
    adapter_->solve(1.0, block_view);

    adapter_->begin_traction(block_view);
    for (std::size_t faultNo = 0, num = adapter_->faultMap().local_size(); faultNo < num;
         ++faultNo) {
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
