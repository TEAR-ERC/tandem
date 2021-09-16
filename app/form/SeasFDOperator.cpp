#include "SeasFDOperator.h"
#include "common/PetscUtil.h"

#include "form/RefElement.h"

namespace tndm {

SeasFDOperator::SeasFDOperator(std::unique_ptr<dg_t> dgop,
                               std::unique_ptr<AbstractAdapterOperator> adapter,
                               std::unique_ptr<AbstractFrictionOperator> friction)
    : dgop_(std::move(dgop)), adapter_(std::move(adapter)), friction_(std::move(friction)),
      traction_(adapter_->traction_block_size(), adapter_->num_local_elements(), adapter_->comm()),
      b_(dgop_->block_size(), dgop_->num_local_elements(), comm()),
      tmp_(dgop_->block_size(), dgop_->num_local_elements(), comm()),
      disp_scatter_(dgop_->topo().elementScatterPlan()),
      disp_ghost_(disp_scatter_.recv_prototype<double>(dgop_->block_size(), ALIGNMENT)),
      state_scatter_(adapter_->fault_map().scatter_plan()),
      state_ghost_(state_scatter_.recv_prototype<double>(friction_->block_size(), ALIGNMENT)) {

    r_dv = profile_.add("dv");
    r_du_apply = profile_.add("du (apply)");
    r_du_other = profile_.add("du (other)");
    r_ds = profile_.add("ds");

    flops_du_apply = dgop_->flops_apply();
}

double SeasFDOperator::cfl_time_step() const {
    double local_dt = dgop_->local_cfl_time_step();
    double global_dt;
    MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, comm());
    return global_dt;
}

void SeasFDOperator::initial_condition(BlockVector& v, BlockVector& u, BlockVector& s) {
    if (u_ini_) {
        dgop_->project((*u_ini_)(), u);
    } else {
        u.set_zero();
    }
    if (v_ini_) {
        dgop_->project((*v_ini_)(), v);
    } else {
        v.set_zero();
    }

    friction_->pre_init(s);
    update_ghost_state(s);
    update_traction(u, s);
    friction_->init(0.0, traction_, s);
}

void SeasFDOperator::rhs(double time, BlockVector const& v, BlockVector const& u,
                         BlockVector const& s, BlockVector& dv, BlockVector& du, BlockVector& ds) {
    profile_.begin(r_dv);
    auto v_handle = v.begin_access_readonly();
    auto du_handle = du.begin_access();
    for (std::size_t elNo = 0, num = dgop_->num_local_elements(); elNo < num; ++elNo) {
        auto v_block = v_handle.subtensor(slice{}, elNo);
        auto du_block = du_handle.subtensor(slice{}, elNo);
        du_block.copy_values(v_block);
    }
    du.end_access(du_handle);
    s.end_access_readonly(v_handle);
    profile_.end(r_dv, flops_dv);

    profile_.begin(r_du_apply);
    dgop_->apply(u, tmp_);
    profile_.end(r_du_apply, flops_du_apply);

    profile_.begin(r_du_other);
    update_ghost_state(s);
    auto state_view = make_state_view(s);
    dgop_->set_slip(adapter_->slip_bc(state_view));
    dgop_->set_dirichlet((*fun_boundary_)(time));

    b_.set_zero();
    dgop_->rhs(b_);
    CHKERRTHROW(VecAXPY(b_.vec(), -1.0, tmp_.vec()));
    dgop_->apply_inverse_mass(b_, dv);

    dgop_->set_slip(invalid_slip_bc());
    profile_.end(r_du_other, flops_du_other);

    profile_.begin(r_ds);
    update_traction(u, s);
    friction_->rhs(time, traction_, s, ds);
    profile_.end(r_ds, flops_ds);
}

void SeasFDOperator::update_traction(BlockVector const& u, BlockVector const& s) {
    disp_scatter_.begin_scatter(u, disp_ghost_);
    disp_scatter_.wait_scatter();
    auto disp_view = LocalGhostCompositeView(u, disp_ghost_);
    auto state_view = make_state_view(s);
    dgop_->set_slip(adapter_->slip_bc(state_view));
    adapter_->traction(disp_view, traction_);
    dgop_->set_slip(invalid_slip_bc());
}

} // namespace tndm
