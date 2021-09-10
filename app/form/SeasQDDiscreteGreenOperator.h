#ifndef SEASQDDISCRETEGREENOPERATOR_20210907_H
#define SEASQDDISCRETEGREENOPERATOR_20210907_H

#include "common/PetscUtil.h"
#include "common/PetscVector.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/FacetFunctionalFactory.h"
#include "form/SeasQDOperator.h"

#include "parallel/LocalGhostCompositeView.h"
#include "util/Stopwatch.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <iostream>
#include <memory>
#include <utility>

namespace tndm {

class SeasQDDiscreteGreenOperator : public SeasQDOperator {
public:
    using base = SeasQDOperator;

    SeasQDDiscreteGreenOperator(std::unique_ptr<typename base::dg_t> dgop,
                                std::unique_ptr<AbstractAdapterOperator> adapter,
                                std::unique_ptr<AbstractFrictionOperator> friction,
                                bool matrix_free = false, MGConfig const& mg_config = MGConfig())
        : base(std::move(dgop), std::move(adapter), std::move(friction), matrix_free, mg_config) {
        compute_discrete_greens_function();
    }
    ~SeasQDDiscreteGreenOperator() { MatDestroy(&G_); }

    void set_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
        base::set_boundary(std::move(fun));
        compute_boundary_traction();
    }

    void initial_condition(BlockVector& state) {
        base::friction().pre_init(state);

        update_traction(0.0, state);

        base::friction().init(base::traction_, state);
    }

    void rhs(double time, BlockVector const& state, BlockVector& result) {
        update_traction(time, state);

        base::friction().rhs(time, base::traction_, state, result);
    }

    void update_internal_state(double time, BlockVector const& state,
                               bool state_changed_since_last_rhs, bool require_traction,
                               bool require_displacement) {
        bool require_solve = state_changed_since_last_rhs && require_traction;
        bool require_solve_domain = require_displacement;
        if (!require_solve && !require_solve_domain) {
            return;
        }

        if (require_solve) {
            update_traction(time, state);
        }

        if (require_solve_domain) {
            base::update_ghost_state(state);
            base::solve(time, base::make_state_view(state));
        }
    }

protected:
    void update_traction(double time, BlockVector const& state) {
        base::update_ghost_state(state);
        auto state_view = base::make_state_view(state);
        for (std::size_t faultNo = 0, num = base::friction().num_local_elements(); faultNo < num;
             ++faultNo) {
            S_->insert_block(faultNo, state_view.get_block(faultNo));
        }
        S_->begin_assembly();
        S_->end_assembly();

        CHKERRTHROW(MatMult(G_, S_->vec(), base::traction_.vec()));
        CHKERRTHROW(VecAXPY(base::traction_.vec(), time, t_boundary_->vec()));
    }

private:
    void compute_discrete_greens_function();
    void compute_boundary_traction();

    Mat G_ = nullptr;
    std::unique_ptr<PetscVector> S_;
    std::unique_ptr<PetscVector> t_boundary_;
};

void SeasQDDiscreteGreenOperator::compute_discrete_greens_function() {
    auto slip_block_size = base::friction().slip_block_size();

    PetscInt num_local_elements = base::adapter().num_local_elements();
    PetscInt m_bs = base::adapter().traction_block_size();
    PetscInt n_bs = 1;
    PetscInt m = num_local_elements * m_bs;
    PetscInt n = num_local_elements * slip_block_size * n_bs;

    MPI_Comm comm = base::comm();

    int rank;
    MPI_Comm_rank(comm, &rank);

    PetscInt mb_offset = 0;
    PetscInt nb_offset = 0;
    MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
    mb_offset -= num_local_elements;
    MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
    nb_offset -= n;

    CHKERRTHROW(MatCreateDense(comm, m, n, PETSC_DECIDE, PETSC_DECIDE, nullptr, &G_));
    CHKERRTHROW(MatSetBlockSizes(G_, m_bs, n_bs));

    S_ = std::make_unique<PetscVector>(slip_block_size, num_local_elements, comm);
    t_boundary_ = std::make_unique<PetscVector>(m_bs, num_local_elements, comm);

    auto scatter = Scatter(base::adapter().fault_map().scatter_plan());
    auto ghost = scatter.template recv_prototype<double>(slip_block_size, ALIGNMENT);

    PetscInt N;
    CHKERRTHROW(VecGetSize(S_->vec(), &N));

    Stopwatch sw;
    double solve_time = 0.0;
    for (PetscInt i = 0; i < N; ++i) {

        if (rank == 0) {
            std::cout << "Computing Green's function " << (i + 1) << "/" << N;
        }
        sw.start();
        CHKERRTHROW(VecZeroEntries(S_->vec()));
        if (i >= nb_offset && i < nb_offset + m) {
            PetscScalar one = 1.0;
            CHKERRTHROW(VecSetValue(S_->vec(), i, one, INSERT_VALUES));
        }
        S_->begin_assembly();
        S_->end_assembly();

        scatter.begin_scatter(*S_, ghost);
        scatter.wait_scatter();

        auto S_view = LocalGhostCompositeView(*S_, ghost);
        base::solve(0.0, S_view);
        base::update_traction(S_view);

        auto traction_handle = base::traction_.begin_access_readonly();
        for (std::size_t faultNo = 0; faultNo < num_local_elements; ++faultNo) {
            PetscInt g_m = mb_offset + faultNo;
            PetscInt g_n = i;
            auto traction_block = traction_handle.subtensor(slice{}, faultNo);
            CHKERRTHROW(
                MatSetValuesBlocked(G_, 1, &g_m, 1, &g_n, traction_block.data(), INSERT_VALUES));
        }
        base::traction_.end_access_readonly(traction_handle);
        solve_time += sw.stop();
        if (rank == 0) {
            constexpr double Days = 3600.0 * 24.0;
            constexpr double Hours = 3600.0;
            constexpr double Minutes = 60.0;
            double avg_time = solve_time / (i + 1);
            double etl = avg_time * (N - i - 1);
            double etl_d = std::floor(etl / Days);
            etl -= etl_d * Days;
            double etl_h = std::floor(etl / Hours);
            etl -= etl_h * Hours;
            double etl_m = std::floor(etl / Minutes);
            etl -= etl_m * Minutes;
            std::cout << " (" << etl_d << "d " << etl_h << "h " << etl_m << "m " << std::floor(etl)
                      << "s left)" << std::endl;
        }
    }

    CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));
}

void SeasQDDiscreteGreenOperator::compute_boundary_traction() {
    MPI_Comm comm = base::comm();
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::cout << "Computing boundary Green's function" << std::endl;
    }

    auto slip_block_size = base::friction().slip_block_size();
    auto scatter = Scatter(base::adapter().fault_map().scatter_plan());
    auto ghost = scatter.template recv_prototype<double>(slip_block_size, ALIGNMENT);

    CHKERRTHROW(VecZeroEntries(S_->vec()));
    scatter.begin_scatter(*S_, ghost);
    scatter.wait_scatter();

    auto S_view = LocalGhostCompositeView(*S_, ghost);
    base::solve(1.0, S_view);
    base::update_traction(S_view);

    CHKERRTHROW(VecCopy(base::traction_.vec(), t_boundary_->vec()));
}

} // namespace tndm

#endif // SEASQDDISCRETEGREENOPERATOR_20210907_H
