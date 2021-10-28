#ifndef SEASQDOPERATOR_20210906_H
#define SEASQDOPERATOR_20210906_H

#include "common/MGConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscVector.h"
#include "config.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/FacetFunctionalFactory.h"

#include "form/AbstractDGOperator.h"
#include "interface/BlockVector.h"
#include "parallel/LocalGhostCompositeView.h"
#include "parallel/Scatter.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Tensor.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

class SeasQDOperator {
public:
    using dg_t = AbstractDGOperator<DomainDimension>;

    SeasQDOperator(std::unique_ptr<dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
                   std::unique_ptr<AbstractFrictionOperator> friction, bool matrix_free = false,
                   MGConfig const& mg_config = MGConfig());

    inline void warmup() { linear_solver_.warmup(); }

    virtual void set_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun);

    inline auto block_sizes() -> std::array<std::size_t, 1> const {
        return {friction_->block_size()};
    }
    inline auto num_local_elements() -> std::array<std::size_t, 1> const {
        return {friction_->num_local_elements()};
    }
    inline MPI_Comm comm() const { return dgop_->topo().comm(); }

    inline AbstractAdapterOperator& adapter() { return *adapter_; }
    inline AbstractAdapterOperator const& adapter() const { return *adapter_; }
    inline dg_t& domain() { return *dgop_; }
    inline dg_t const& domain() const { return *dgop_; }
    inline AbstractFrictionOperator& friction() { return *friction_; }
    inline AbstractFrictionOperator const& friction() const { return *friction_; }

    void initial_condition(BlockVector& state);
    void rhs(double time, BlockVector const& state, BlockVector& result);
    void update_internal_state(double time, BlockVector const& state,
                               bool state_changed_since_last_rhs, bool require_traction,
                               bool require_displacement);

    inline auto displacement(std::vector<std::size_t> const& subset) const {
        return dgop_->solution(linear_solver_.x(), subset);
    }
    inline auto displacement() const { return dgop_->solution(linear_solver_.x()); }

    inline auto state(double time, BlockVector const& state_vec,
                      std::vector<std::size_t> const& subset) {
        return friction_->state(time, traction_, state_vec, subset);
    }
    inline auto state(double time, BlockVector const& state_vec) {
        return friction_->state(time, traction_, state_vec);
    }

protected:
    inline auto invalid_slip_bc() {
        return [](std::size_t, Matrix<double>&, bool) {
            throw std::logic_error("Slip boundary condition not set");
        };
    }

    inline void update_ghost_state(BlockVector const& state) {
        state_scatter_.begin_scatter(state, state_ghost_);
        state_scatter_.wait_scatter();
    }

    inline auto make_state_view(BlockVector const& state) -> LocalGhostCompositeView {
        return LocalGhostCompositeView(state, state_ghost_);
    }

    void solve(double time, BlockView const& state_view);
    void update_traction(BlockView const& state_view);

private:
    std::unique_ptr<dg_t> dgop_;
    PetscLinearSolver linear_solver_;
    std::unique_ptr<AbstractAdapterOperator> adapter_;
    std::unique_ptr<AbstractFrictionOperator> friction_;

    Scatter disp_scatter_;
    SparseBlockVector<double> disp_ghost_;

    Scatter state_scatter_;
    SparseBlockVector<double> state_ghost_;

    std::unique_ptr<AbstractFacetFunctionalFactory> fun_boundary_ = nullptr;

protected:
    PetscVector traction_;
};

} // namespace tndm

#endif // SEASQDOPERATOR_20210906_H
