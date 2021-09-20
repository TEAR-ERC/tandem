#ifndef SEASFDOPERATOR_20210907_H
#define SEASFDOPERATOR_20210907_H

#include "common/MGConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscVector.h"
#include "config.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/FacetFunctionalFactory.h"
#include "form/FacetInfo.h"
#include "form/VolumeFunctionalFactory.h"

#include "form/AbstractDGOperator.h"
#include "form/BoundaryMap.h"
#include "interface/BlockVector.h"
#include "parallel/LocalGhostCompositeView.h"
#include "parallel/Profile.h"
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

class SeasFDOperator {
public:
    using dg_t = AbstractDGOperator<DomainDimension>;

    SeasFDOperator(std::unique_ptr<dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
                   std::unique_ptr<AbstractFrictionOperator> friction);

    inline void set_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
        fun_boundary_ = std::move(fun);
    }
    inline void set_initial_displacement(std::unique_ptr<AbstractVolumeFunctionalFactory> u_ini) {
        u_ini_ = std::move(u_ini);
    }
    inline void set_initial_velocity(std::unique_ptr<AbstractVolumeFunctionalFactory> v_ini) {
        v_ini_ = std::move(v_ini);
    }

    inline auto block_sizes() -> std::array<std::size_t, 3> const {
        return {dgop_->block_size(), dgop_->block_size(), friction_->block_size()};
    }
    inline auto num_local_elements() -> std::array<std::size_t, 3> const {
        return {dgop_->num_local_elements(), dgop_->num_local_elements(),
                friction_->num_local_elements()};
    }
    inline MPI_Comm comm() const { return dgop_->topo().comm(); }

    inline AbstractAdapterOperator& adapter() { return *adapter_; }
    inline AbstractAdapterOperator const& adapter() const { return *adapter_; }
    inline dg_t& domain() { return *dgop_; }
    inline dg_t const& domain() const { return *dgop_; }
    inline AbstractFrictionOperator& friction() { return *friction_; }
    inline AbstractFrictionOperator const& friction() const { return *friction_; }

    double cfl_time_step() const;
    void initial_condition(BlockVector& v, BlockVector& u, BlockVector& s);
    void rhs(double time, BlockVector const& v, BlockVector const& u, BlockVector const& s,
             BlockVector& dv, BlockVector& du, BlockVector& ds);

    auto domain_function(BlockVector const& x, std::vector<std::size_t> const& subset) const {
        return dgop_->solution(x, subset);
    }
    inline auto domain_function(BlockVector const& x) const { return dgop_->solution(x); }

    inline auto fault_state(double time, BlockVector const& state_vec,
                            std::vector<std::size_t> const& subset) {
        return friction_->state(time, traction_, state_vec, subset);
    }
    inline auto fault_state(double time, BlockVector const& state_vec) {
        return friction_->state(time, traction_, state_vec);
    }

    inline Profile const& profile() const { return profile_; };

private:
    inline auto invalid_slip_bc() {
        return [](std::size_t, Matrix<double>&, bool) {
            throw std::logic_error("Slip boundary condition not set");
        };
    }

    inline auto make_state_view(BlockVector const& state) -> LocalGhostCompositeView {
        return LocalGhostCompositeView(state, state_ghost_);
    }

    void update_traction(BlockVector const& u, BlockVector const& s);

    std::unique_ptr<dg_t> dgop_;
    std::unique_ptr<AbstractAdapterOperator> adapter_;
    std::unique_ptr<AbstractFrictionOperator> friction_;

    PetscVector traction_;
    Scatter disp_scatter_;
    SparseBlockVector<double> disp_ghost_;

    Scatter state_scatter_;
    SparseBlockVector<double> state_ghost_;

    std::unique_ptr<AbstractFacetFunctionalFactory> fun_boundary_ = nullptr;
    std::unique_ptr<AbstractVolumeFunctionalFactory> u_ini_ = nullptr;
    std::unique_ptr<AbstractVolumeFunctionalFactory> v_ini_ = nullptr;

    Profile profile_;
    std::size_t r_dv, r_du, r_ds;
    uint64_t flops_dv = 0, flops_du = 0, flops_ds = 0;
};

} // namespace tndm

#endif // SEASFDOPERATOR_20210907_H
