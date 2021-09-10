#ifndef SEASFDOPERATOR_20210907_H
#define SEASFDOPERATOR_20210907_H

#include "common/MGConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscUtil.h"
#include "common/PetscVector.h"
#include "config.h"
#include "form/AbstractFrictionOperator.h"

#include "form/BoundaryMap.h"
#include "interface/BlockVector.h"
#include "parallel/LocalGhostCompositeView.h"
#include "parallel/Scatter.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "util/Scratch.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <petscdm.h>
#include <stdexcept>
#include <utility>

namespace tndm {

template <typename AdapterOperator, typename DomainOperator> class SeasFDOperator {
public:
    constexpr static std::size_t Dim = DomainOperator::local_operator_t::Dim;
    constexpr static std::size_t DomainNumQuantities =
        DomainOperator::local_operator_t::NumQuantities;
    using time_functional_t =
        std::function<std::array<double, DomainNumQuantities>(std::array<double, Dim + 1u> const&)>;
    using vector_functional_t = typename DomainOperator::vector_functional_t;

    SeasFDOperator(std::unique_ptr<DomainOperator> dgop, std::unique_ptr<AdapterOperator> adapter,
                   std::unique_ptr<AbstractFrictionOperator> friction)
        : dgop_(std::move(dgop)), adapter_(std::move(adapter)), friction_(std::move(friction)),
          traction_(adapter_->lop().traction_block_size(), adapter_->num_local_elements(),
                    adapter_->comm()),
          b_(dgop_->block_size(), dgop_->num_local_elements(), comm()),
          tmp_(dgop_->block_size(), dgop_->num_local_elements(), comm()),
          disp_scatter_(dgop_->topo().elementScatterPlan()),
          disp_ghost_(
              disp_scatter_.recv_prototype<double>(dgop_->block_size(), dgop_->lop().alignment())),
          state_scatter_(adapter_->fault_map().scatter_plan()),
          state_ghost_(state_scatter_.recv_prototype<double>(friction_->block_size(), ALIGNMENT)) {}

    void set_boundary(time_functional_t fun) { fun_boundary_ = std::move(fun); }
    void set_initial_displacement(vector_functional_t u_ini) { u_ini_ = std::move(u_ini); }
    void set_initial_velocity(vector_functional_t v_ini) { v_ini_ = std::move(v_ini); }

    auto block_sizes() -> std::array<std::size_t, 3> const {
        return {dgop_->block_size(), dgop_->block_size(), friction_->block_size()};
    }
    auto num_local_elements() -> std::array<std::size_t, 3> const {
        return {dgop_->num_local_elements(), dgop_->num_local_elements(),
                friction_->num_local_elements()};
    }
    MPI_Comm comm() const { return dgop_->topo().comm(); }

    AdapterOperator& adapter() { return *adapter_; }
    AdapterOperator const& adapter() const { return *adapter_; }
    DomainOperator& domain() { return *dgop_; }
    DomainOperator const& domain() const { return *dgop_; }
    AbstractFrictionOperator& friction() { return *friction_; }
    AbstractFrictionOperator const& friction() const { return *friction_; }

    void initial_condition(BlockVector& v, BlockVector& u, BlockVector& s) {
        dgop_->project(u_ini_, u);
        dgop_->project(v_ini_, v);

        friction_->pre_init(s);
        update_ghost_state(s);
        update_traction(u, s);
        friction_->init(traction_, s);
    }

    void rhs(double time, BlockVector const& v, BlockVector const& u, BlockVector const& s,
             BlockVector& dv, BlockVector& du, BlockVector& ds) {
        auto v_handle = v.begin_access_readonly();
        auto du_handle = du.begin_access();
        for (std::size_t elNo = 0, num = dgop_->num_local_elements(); elNo < num; ++elNo) {
            auto v_block = v_handle.subtensor(slice{}, elNo);
            auto du_block = du_handle.subtensor(slice{}, elNo);
            du_block.copy_values(v_block);
        }
        du.end_access(du_handle);
        s.end_access_readonly(v_handle);

        update_ghost_state(s);
        auto state_view = make_state_view(s);
        dgop_->lop().set_slip(adapter_->slip_bc(state_view));
        dgop_->lop().set_dirichlet(
            [this, time](std::array<double, Dim> const& x) {
                std::array<double, Dim + 1u> xt;
                std::copy(x.begin(), x.end(), xt.begin());
                xt.back() = time;
                return this->fun_boundary_(xt);
            },
            adapter_->lop().ref_normal());

        b_.set_zero();
        dgop_->rhs(b_);
        dgop_->apply(u, tmp_);
        CHKERRTHROW(VecAXPY(b_.vec(), -1.0, tmp_.vec()));
        dgop_->apply_inverse_mass(b_, dv);

        dgop_->lop().set_slip(invalid_slip_bc());

        update_traction(u, s);
        friction_->rhs(time, traction_, s, ds);
    }

    template <typename Iterator>
    auto domain_function(BlockVector const& x, Iterator first, Iterator last) const {
        return dgop_->solution(x, std::forward<Iterator>(first), std::forward<Iterator>(last));
    }
    auto domain_function(BlockVector const& x) const { return dgop_->solution(x); }

    auto fault_state(BlockVector const& state_vec, std::vector<std::size_t> const& subset) {
        return friction_->state(traction_, state_vec, subset);
    }
    auto fault_state(BlockVector const& state_vec) {
        return friction_->state(traction_, state_vec);
    }

private:
    auto invalid_slip_bc() {
        return [](std::size_t, Matrix<double>&, bool) {
            throw std::logic_error("Slip boundary condition not set");
        };
    }

    void update_ghost_state(BlockVector const& state) {
        state_scatter_.begin_scatter(state, state_ghost_);
        state_scatter_.wait_scatter();
    }

    auto make_state_view(BlockVector const& state) {
        return LocalGhostCompositeView(state, state_ghost_);
    }

    void update_traction(BlockVector const& u, BlockVector const& s) {
        disp_scatter_.begin_scatter(u, disp_ghost_);
        disp_scatter_.wait_scatter();
        auto disp_view = LocalGhostCompositeView(u, disp_ghost_);
        auto state_view = make_state_view(s);
        dgop_->lop().set_slip(adapter_->slip_bc(state_view));
        adapter_->traction(disp_view, traction_);
        dgop_->lop().set_slip(invalid_slip_bc());
    }

    std::unique_ptr<DomainOperator> dgop_;
    std::unique_ptr<AdapterOperator> adapter_;
    std::unique_ptr<AbstractFrictionOperator> friction_;

    PetscVector traction_;
    PetscVector b_, tmp_;
    Scatter disp_scatter_;
    SparseBlockVector<double> disp_ghost_;

    Scatter state_scatter_;
    SparseBlockVector<double> state_ghost_;

    time_functional_t fun_boundary_ =
        [](std::array<double, Dim + 1u> const& x) -> std::array<double, DomainNumQuantities> {
        return {};
    };

    vector_functional_t u_ini_ =
        [](std::array<double, Dim> const& x) -> std::array<double, DomainNumQuantities> {
        return {};
    };

    vector_functional_t v_ini_ =
        [](std::array<double, Dim> const& x) -> std::array<double, DomainNumQuantities> {
        return {};
    };
};

} // namespace tndm

#endif // SEASFDOPERATOR_20210907_H
