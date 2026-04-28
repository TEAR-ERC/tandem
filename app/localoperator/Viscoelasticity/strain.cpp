/*
Viscoelasticity strain history and displacement storage

Strain history: initialisation, displacement storage, deviatoric and partial
strain computation at volume and facet quad points, and history buffer
rotation (old <= new).
*/
#include "Viscoelasticity.h"

#include "kernels/viscoelasticity/init.h"
#include "kernels/viscoelasticity/kernel.h"
#include "kernels/viscoelasticity/tensor.h"

#include <cmath>
#include <stdexcept>

namespace tensor = tndm::viscoelasticity::tensor;
namespace init = tndm::viscoelasticity::init;
namespace kernel = tndm::viscoelasticity::kernel;

namespace tndm {

// Strain history initialisation (called once at t=0 by DGOperator constructor)
void Viscoelasticity::initialize_strain_tensor_Q(std::size_t elNo) {
    // Zero all per-element strain history buffers at volume quad points.
    // Called once per element before the first time step.
    for (std::size_t q = 0; q < volRule.size(); ++q) {
        auto fill_q = [q](auto& storage) { std::fill(storage[q].begin(), storage[q].end(), 0.0); };
        fill_q(strainHistory_Q[elNo].get<partial_strain_old_Q>());
        fill_q(strainHistory_Q[elNo].get<partial_strain_new_Q>());
        fill_q(strainHistory_Q[elNo].get<deviatoric_strain_old_Q>());
        fill_q(strainHistory_Q[elNo].get<deviatoric_strain_new_Q>());
        fill_q(strainHistory_Q[elNo].get<total_strain_Q>());
        fill_q(strainHistory_Q[elNo].get<trace_tensor_Q>());
        strainHistory_Q[elNo].get<strain_trace_Q>()[q] = 0.0;
    }
}

void Viscoelasticity::initialize_strain_tensor_q(std::size_t fctNo) {
    // Zero all per-facet strain history buffers at facet quad points.
    for (std::size_t q = 0; q < fctRule.size(); ++q) {
        auto fill_q = [q](auto& storage) { std::fill(storage[q].begin(), storage[q].end(), 0.0); };
        // Side 0
        fill_q(strainHistory_q[fctNo].get<partial_strain_old_q_0>());
        fill_q(strainHistory_q[fctNo].get<partial_strain_new_q_0>());
        fill_q(strainHistory_q[fctNo].get<deviatoric_strain_old_q_0>());
        fill_q(strainHistory_q[fctNo].get<deviatoric_strain_new_q_0>());
        fill_q(strainHistory_q[fctNo].get<total_strain_q_0>());
        fill_q(strainHistory_q[fctNo].get<trace_tensor_q_0>());
        // Side 1
        fill_q(strainHistory_q[fctNo].get<partial_strain_old_q_1>());
        fill_q(strainHistory_q[fctNo].get<partial_strain_new_q_1>());
        fill_q(strainHistory_q[fctNo].get<deviatoric_strain_old_q_1>());
        fill_q(strainHistory_q[fctNo].get<deviatoric_strain_new_q_1>());
        fill_q(strainHistory_q[fctNo].get<total_strain_q_1>());
        fill_q(strainHistory_q[fctNo].get<trace_tensor_q_1>());
        // Scalars
        fill_q(strainHistory_q[fctNo].get<average_traction>());
        strainHistory_q[fctNo].get<strain_trace_q_0>()[q] = 0.0;
        strainHistory_q[fctNo].get<strain_trace_q_1>()[q] = 0.0;
        strainHistory_q[fctNo].get<traction_history_normal>()[q] = 0.0;
    }
}

void Viscoelasticity::initialize_displacement_field(std::size_t elNo) {
    for (std::size_t i = 0; i < space_.numBasisFunctions(); ++i) {
        std::fill(displacementField[elNo].get<displacement_field>()[i].begin(),
                  displacementField[elNo].get<displacement_field>()[i].end(), 0.0);
    }
}

// Displacement storage
void Viscoelasticity::store_displacement_field(std::size_t elNo, const double* U_data) const {
    const std::size_t num_basis = space_.numBasisFunctions();
    auto disp = displacementField[elNo].get<displacement_field>();
    for (std::size_t i = 0; i < num_basis; ++i) {
        for (std::size_t d = 0; d < Dim; ++d) {
            const double val = U_data[i * Dim + d];
#ifndef NDEBUG
            if (std::isnan(val)) {
                throw std::runtime_error("VE: NaN in displacement field at element " +
                                         std::to_string(elNo) + " dof " + std::to_string(i) +
                                         " dim " + std::to_string(d));
            }
#endif
            disp[i][d] = val;
        }
    }
}

/*
Deviatoric strain computation — volume and facet

Four-kernel pipeline per element:
  1. strain_tensor_Q      => ε_ij = (1/2)(∂u_i/∂x_j + ∂u_j/∂x_i)
  2. strain_trace_Q       => Tr(ε) = ε_kk
  3. trace_tensor_Q       => Tr(ε) δ_ij
  4. deviatoric_strain_Q  => ε^dev = ε − (1/dim) Tr(ε) I

Result stored in deviatoric_strain_new_Q (not overwriting _old yet).
The swap _old <= _new happens in update_deviatoric_strain_Q.
*/
void Viscoelasticity::compute_deviatoric_strain_Q(std::size_t elNo) {
    alignas(ALIGNMENT) double Dx_Q[tensor::Dx_Q::size()];
    {
        kernel::Dx_Q dxKrnl;
        dxKrnl.Dx_Q = Dx_Q;
        dxKrnl.Dxi_Q = Dxi_Q.data();
        dxKrnl.G = vol[elNo].get<JInv>().data()->data();
        dxKrnl.execute();
    }

    auto& hist = strainHistory_Q[elNo];
    double* total_strain = hist.get<total_strain_Q>().data()->data();
    double* strain_trace = hist.get<strain_trace_Q>().data();
    double* trace_tensor = hist.get<trace_tensor_Q>().data()->data();
    double* dev_new = hist.get<deviatoric_strain_new_Q>().data()->data();
    const double* U_data = displacementField[elNo].get<displacement_field>().data()->data();

    kernel::strain_tensor_Q krnl_strain;
    krnl_strain.strain_tensor_Q = total_strain;
    krnl_strain.Dx_Q = Dx_Q;
    krnl_strain.U = U_data;
    krnl_strain.execute();

    kernel::strain_trace_Q krnl_trace;
    krnl_trace.strain_trace_Q = strain_trace;
    krnl_trace.strain_tensor_Q = total_strain;
    krnl_trace.delta = init::delta::Values;
    krnl_trace.execute();

    kernel::trace_tensor_Q krnl_trace_tensor;
    krnl_trace_tensor.trace_tensor_Q = trace_tensor;
    krnl_trace_tensor.strain_trace_Q = strain_trace;
    krnl_trace_tensor.delta = init::delta::Values;
    krnl_trace_tensor.execute();

    kernel::deviatoric_strain_tensor_Q krnl_dev;
    krnl_dev.deviatoric_strain_tensor_Q = dev_new;
    krnl_dev.strain_tensor_Q = total_strain;
    krnl_dev.trace_tensor_Q = trace_tensor;
    krnl_dev.execute();
}

void Viscoelasticity::compute_deviatoric_strain_q(std::size_t fctNo, FacetInfo const& info) {
    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];
    alignas(ALIGNMENT) double Dx_q1[tensor::Dx_q::size(1)];
    {
        kernel::Dx_q dxKrnl;
        dxKrnl.Dx_q(0) = Dx_q0;
        dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
        dxKrnl.Dx_q(1) = Dx_q1;
        dxKrnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
        dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
        dxKrnl.Dxi_q(1) = Dxi_q[info.localNo[1]].data();
        dxKrnl.execute(0);
        dxKrnl.execute(1);
    }

    // Run the four-kernel pipeline for each side
    for (int side = 0; side < 2; ++side) {
        const double* Dx_qs = (side == 0) ? Dx_q0 : Dx_q1;
        const double* U_data =
            displacementField[info.up[side]].get<displacement_field>().data()->data();
        auto& hist = strainHistory_q[fctNo];

        double* total_q = (side == 0) ? hist.get<total_strain_q_0>().data()->data()
                                      : hist.get<total_strain_q_1>().data()->data();
        double* trace_q =
            (side == 0) ? hist.get<strain_trace_q_0>().data() : hist.get<strain_trace_q_1>().data();
        double* traceT_q = (side == 0) ? hist.get<trace_tensor_q_0>().data()->data()
                                       : hist.get<trace_tensor_q_1>().data()->data();
        double* dev_new_q = (side == 0) ? hist.get<deviatoric_strain_new_q_0>().data()->data()
                                        : hist.get<deviatoric_strain_new_q_1>().data()->data();

        kernel::strain_tensor_q krnl_strain;
        krnl_strain.strain_tensor_q(side) = total_q;
        krnl_strain.Dx_q(side) = Dx_qs;
        krnl_strain.U = U_data;
        krnl_strain.execute(side);

        kernel::strain_trace_q krnl_trace;
        krnl_trace.strain_trace_q(side) = trace_q;
        krnl_trace.strain_tensor_q(side) = total_q;
        krnl_trace.delta = init::delta::Values;
        krnl_trace.execute(side);

        kernel::trace_tensor_q krnl_traceT;
        krnl_traceT.trace_tensor_q(side) = traceT_q;
        krnl_traceT.strain_trace_q(side) = trace_q;
        krnl_traceT.delta = init::delta::Values;
        krnl_traceT.execute(side);

        kernel::deviatoric_strain_tensor_q krnl_dev;
        krnl_dev.deviatoric_strain_tensor_q(side) = dev_new_q;
        krnl_dev.strain_tensor_q(side) = total_q;
        krnl_dev.trace_tensor_q(side) = traceT_q;
        krnl_dev.execute(side);
    }
}

/*
Partial strain recurrence — volume and facet

qⁿ⁺¹ = qⁿ exp(-Δt/τ) + g(Δt)(ε^{dev,n+1} − ε^{dev,n})

Reads: old_partial (_old), old deviatoric (_old), new deviatoric (_new)
Writes: new_partial (_new)
The swap _old <= _new happens in update_partial_strain_Q.
*/
void Viscoelasticity::compute_partial_strain_Q(std::size_t elNo) {
    auto& hist = strainHistory_Q[elNo];
    kernel::partialStrain_Q ps;
    ps.old_partial_strain_tensor_Q = hist.get<partial_strain_old_Q>().data()->data();
    ps.new_partial_strain_tensor_Q = hist.get<partial_strain_new_Q>().data()->data();
    ps.old_deviatoric_strain_tensor_Q = hist.get<deviatoric_strain_old_Q>().data()->data();
    ps.new_deviatoric_strain_tensor_Q = hist.get<deviatoric_strain_new_Q>().data()->data();
    ps.ratio_Q = volPre[elNo].get<ratio_Q>().data();
    ps.g_dt_Q = volPre[elNo].get<g_dt_Q>().data();
    ps.execute();
}

void Viscoelasticity::compute_partial_strain_q(std::size_t fctNo, FacetInfo const& info) {
    auto& hist = strainHistory_q[fctNo];
    kernel::partialStrain_q ps;
    ps.ratio_q = fctPre[fctNo].get<ratio_q>().data();
    ps.g_dt_q = fctPre[fctNo].get<g_dt_q>().data();
    for (int side = 0; side < 2; ++side) {
        ps.old_partial_strain_tensor_q(side) =
            (side == 0) ? hist.get<partial_strain_old_q_0>().data()->data()
                        : hist.get<partial_strain_old_q_1>().data()->data();
        ps.new_partial_strain_tensor_q(side) =
            (side == 0) ? hist.get<partial_strain_new_q_0>().data()->data()
                        : hist.get<partial_strain_new_q_1>().data()->data();
        ps.old_deviatoric_strain_tensor_q(side) =
            (side == 0) ? hist.get<deviatoric_strain_old_q_0>().data()->data()
                        : hist.get<deviatoric_strain_old_q_1>().data()->data();
        ps.new_deviatoric_strain_tensor_q(side) =
            (side == 0) ? hist.get<deviatoric_strain_new_q_0>().data()->data()
                        : hist.get<deviatoric_strain_new_q_1>().data()->data();
        ps.execute(side);
    }
}

/*
History buffer rotation — old <= new

Must be called AFTER compute_deviatoric/partial_strain and AFTER any stress
output that needs the old values (see Algorithm 1 ordering).
*/
void Viscoelasticity::update_deviatoric_strain_Q(std::size_t elNo) {
    auto& hist = strainHistory_Q[elNo];
    kernel::updateStrain_Q krnl;
    krnl.old_strain_tensor_Q = hist.get<deviatoric_strain_old_Q>().data()->data();
    krnl.new_strain_tensor_Q = hist.get<deviatoric_strain_new_Q>().data()->data();
    krnl.execute();
}

void Viscoelasticity::update_deviatoric_strain_q(std::size_t fctNo) {
    auto& hist = strainHistory_q[fctNo];
    kernel::updateStrain_q krnl;
    for (int side = 0; side < 2; ++side) {
        krnl.old_strain_tensor_q(side) = (side == 0)
                                             ? hist.get<deviatoric_strain_old_q_0>().data()->data()
                                             : hist.get<deviatoric_strain_old_q_1>().data()->data();
        krnl.new_strain_tensor_q(side) = (side == 0)
                                             ? hist.get<deviatoric_strain_new_q_0>().data()->data()
                                             : hist.get<deviatoric_strain_new_q_1>().data()->data();
        krnl.execute(side);
    }
}

void Viscoelasticity::update_partial_strain_Q(std::size_t elNo) {
    auto& hist = strainHistory_Q[elNo];
    kernel::updateStrain_Q krnl;
    krnl.old_strain_tensor_Q = hist.get<partial_strain_old_Q>().data()->data();
    krnl.new_strain_tensor_Q = hist.get<partial_strain_new_Q>().data()->data();
    krnl.execute();
}

void Viscoelasticity::update_partial_strain_q(std::size_t fctNo) {
    auto& hist = strainHistory_q[fctNo];
    kernel::updateStrain_q krnl;
    for (int side = 0; side < 2; ++side) {
        krnl.old_strain_tensor_q(side) = (side == 0)
                                             ? hist.get<partial_strain_old_q_0>().data()->data()
                                             : hist.get<partial_strain_old_q_1>().data()->data();
        krnl.new_strain_tensor_q(side) = (side == 0)
                                             ? hist.get<partial_strain_new_q_0>().data()->data()
                                             : hist.get<partial_strain_new_q_1>().data()->data();
        krnl.execute(side);
    }
}

} // namespace tndm
