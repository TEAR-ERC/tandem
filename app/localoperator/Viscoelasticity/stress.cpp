/*
Total viscoelastic stress computation for VTU output.
Computes σ = σ̂ⁿ⁺¹ + σ̂ⁿ at nodal points via L2 projection.
Expensive - only called when output required.
*/

#include "Viscoelasticity.h"

#include "kernels/viscoelasticity/init.h"
#include "kernels/viscoelasticity/kernel.h"
#include "kernels/viscoelasticity/tensor.h"
#include "util/LinearAllocator.h"

#include <Eigen/LU>
#include <cassert>

namespace tensor = tndm::viscoelasticity::tensor;
namespace init = tndm::viscoelasticity::init;
namespace kernel = tndm::viscoelasticity::kernel;

namespace tndm {

FiniteElementFunction<DomainDimension>
Viscoelasticity::stress_prototype(std::size_t numLocalElements) const {
    // Return stress tensor components as separate scalar fields
    // 3D: σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz (6 components)
    // 2D: σ_xx, σ_yy, σ_xy (3 components)
    std::vector<std::string> names;
    if constexpr (Dim == 3) {
        names = {"stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_xz", "stress_yz"};
    } else {
        names = {"stress_xx", "stress_yy", "stress_xy"};
    }
    return FiniteElementFunction<DomainDimension>(space_.clone(), names, numLocalElements);
}

void Viscoelasticity::stress_volume(std::size_t elNo, Matrix<double>& stress_components,
                                    LinearAllocator<double>&) const {
    // Compute total viscoelastic stress on-the-fly for element elNo
    // Total stress: σ = A_dt * δ_ij * ε_kk + 2 * B_dt * ε_ij + 2*μ₁*ratio*q_ij_old -
    // 2*μ₁*g_dt*ε^{dev}_old where:
    //   A_dt = λ + (2/3) * (μ₁ - μ₁ * g(Δt))   [effective λ, unscaled]
    //   B_dt = μ₀ + μ₁ g(Δt)                   [effective μ, unscaled]
    //   ratio = exp(-Δt/τ)
    //   g_dt = g(Δt)
    //   q_ij_old = old partial strain (internal variable from previous step)
    //   ε^{dev}_old = old deviatoric strain

    std::size_t numBasis = space_.numBasisFunctions();
    std::size_t numQuad = volRule.size();

    // Step 1: Interpolate unscaled material parameters to quadrature points
    // lam, mu0, mu1 are stored at basis function positions
    alignas(ALIGNMENT) double lam_Q[numQuad];
    alignas(ALIGNMENT) double mu0_Q[numQuad];
    alignas(ALIGNMENT) double mu1_Q[numQuad];

    auto const& mat = material[elNo];
    auto const coeff_lam = mat.get<lam>();
    auto const coeff_mu0 = mat.get<mu0>();
    auto const coeff_mu1 = mat.get<mu1>();

    // Interpolate: param_Q = E_Q^T * param_basis
    for (std::size_t q = 0; q < numQuad; ++q) {
        lam_Q[q] = 0.0;
        mu0_Q[q] = 0.0;
        mu1_Q[q] = 0.0;
        for (std::size_t k = 0; k < numBasis; ++k) {
            double E_val = E_Q(k, q);
            lam_Q[q] += E_val * coeff_lam[k];
            mu0_Q[q] += E_val * coeff_mu0[k];
            mu1_Q[q] += E_val * coeff_mu1[k];
        }
    }

    // Step 2: Reuse strain data computed in strain.cpp
    // total_strain_Q and strain_trace_Q are produced by compute_deviatoric_strain_Q().
    auto const& hist = strainHistory_Q[elNo];
    const double* strain_tensor = hist.get<total_strain_Q>().data()->data();
    const double* strain_trace = hist.get<strain_trace_Q>().data();

    // Temporary storage for intermediate stress at quadrature points
    alignas(ALIGNMENT) double stress_total[tensor::stress_total_Q::size()];

    // Step 3: Get time integration scalars and old strain history
    const double* g_dt_Q_data = volPre[elNo].get<g_dt_Q>().data();
    const double* ratio_Q_data = volPre[elNo].get<ratio_Q>().data();

    const double* old_partial_strain = hist.get<partial_strain_old_Q>().data()->data();
    const double* old_dev_strain = hist.get<deviatoric_strain_old_Q>().data()->data();

    // Step 4: Compute unscaled effective Lamé parameters A_dt and B_dt
    alignas(ALIGNMENT) double A_dt_unscaled[numQuad];
    alignas(ALIGNMENT) double B_dt_unscaled[numQuad];
    {
        kernel::precomputeVolumeABUnscaled krnl;
        krnl.A_dt_unscaled = A_dt_unscaled;
        krnl.B_dt_unscaled = B_dt_unscaled;
        krnl.lam_Q = lam_Q;
        krnl.mu0_Q = mu0_Q;
        krnl.mu1_Q = mu1_Q;
        krnl.g_dt_Q = g_dt_Q_data;
        krnl.execute();
    }

    // Step 5: Compute total stress:
    // σ_ij = A_dt * δ_ij * ε_kk + B_dt * ε_ij + 2*μ₁*ratio*q_ij_old - 2*μ₁*g_dt*ε^{dev}_old
    {
        kernel::computeTotalStress krnl;
        krnl.stress_total_Q = stress_total;
        krnl.A_dt_unscaled = A_dt_unscaled;
        krnl.B_dt_unscaled = B_dt_unscaled;
        krnl.strain_trace_Q = strain_trace;
        krnl.strain_tensor_Q = strain_tensor;
        krnl.mu1_Q = mu1_Q;
        krnl.ratio_Q = ratio_Q_data;
        krnl.old_partial_strain_tensor_Q = old_partial_strain;
        krnl.g_dt_Q = g_dt_Q_data;
        krnl.old_deviatoric_strain_tensor_Q = old_dev_strain;
        krnl.delta = init::delta::Values;
        krnl.execute();
    }

    // Step 6: Project stress to nodal basis — same pattern as prepare_volume
    alignas(ALIGNMENT) double Mmem[tensor::matM::size()];
    {
        kernel::project_material_lhs krnl_lhs;
        krnl_lhs.matE_Q_T = matE_Q_T.data();
        krnl_lhs.J = vol[elNo].get<AbsDetJ>().data();
        krnl_lhs.matM = Mmem;
        krnl_lhs.W = volRule.weights().data();
        krnl_lhs.execute();
    }

    alignas(ALIGNMENT) double stress_nodal_rhs[tensor::stress_nodal::size()] = {};
    {
        kernel::projectStressRHS krnl;
        krnl.stress_nodal = stress_nodal_rhs;
        krnl.stress_total_Q = stress_total;
        krnl.matE_Q_T = matE_Q_T.data(); // ← must use matE_Q_T, not E_Q
        krnl.W = volRule.weights().data();
        krnl.J = vol[elNo].get<AbsDetJ>().data();
        krnl.execute();
    }

    // Solve with matM, using the same Map type as prepare_volume
    using MMap = Eigen::Map<Eigen::Matrix<double, tensor::matM::Shape[0], tensor::matM::Shape[1]>,
                            Eigen::Unaligned,
                            Eigen::OuterStride<init::matM::Stop[0] - init::matM::Start[0]>>;
    auto proj = MMap(Mmem).fullPivLu();

    // One solve per stress component (i,j)
    alignas(ALIGNMENT) double stress_nodal_raw[tensor::stress_nodal::size()] = {};
    for (std::size_t j = 0; j < Dim; ++j) {
        for (std::size_t i = 0; i < Dim; ++i) {
            std::size_t offset = i * numBasis + j * numBasis * Dim;
            using Vec = Eigen::Map<Eigen::Matrix<double, tensor::stress_nodal::Shape[0], 1>,
                                   Eigen::Unaligned, Eigen::InnerStride<1>>;
            Vec coeff(stress_nodal_raw + offset);
            Vec rhs(stress_nodal_rhs + offset);
            coeff = proj.solve(rhs);
        }
    }

    // Step 7: Extract symmetric components for output
    // Output layout: stress_components(basis_idx, component_idx)
    // Component ordering: xx, yy, [zz,] xy, [xz, yz]
    //
    // stress_nodal is column-major: (Nbf, Dim, Dim)
    // Element [k][i][j] is at: k + i * Nbf + j * Nbf * Dim
    if constexpr (Dim == 3) {
        assert(6 == stress_components.shape(1));
        for (std::size_t k = 0; k < numBasis; ++k) {
            stress_components(k, 0) =
                stress_nodal_raw[k + 0 * numBasis + 0 * numBasis * Dim]; // σ_xx [k][0][0]
            stress_components(k, 1) =
                stress_nodal_raw[k + 1 * numBasis + 1 * numBasis * Dim]; // σ_yy [k][1][1]
            stress_components(k, 2) =
                stress_nodal_raw[k + 2 * numBasis + 2 * numBasis * Dim]; // σ_zz [k][2][2]
            stress_components(k, 3) =
                stress_nodal_raw[k + 0 * numBasis + 1 * numBasis * Dim]; // σ_xy [k][0][1]
            stress_components(k, 4) =
                stress_nodal_raw[k + 0 * numBasis + 2 * numBasis * Dim]; // σ_xz [k][0][2]
            stress_components(k, 5) =
                stress_nodal_raw[k + 1 * numBasis + 2 * numBasis * Dim]; // σ_yz [k][1][2]
        }
    } else {
        assert(3 == stress_components.shape(1));
        for (std::size_t k = 0; k < numBasis; ++k) {
            stress_components(k, 0) =
                stress_nodal_raw[k + 0 * numBasis + 0 * numBasis * Dim]; // σ_xx [k][0][0]
            stress_components(k, 1) =
                stress_nodal_raw[k + 1 * numBasis + 1 * numBasis * Dim]; // σ_yy [k][1][1]
            stress_components(k, 2) =
                stress_nodal_raw[k + 0 * numBasis + 1 * numBasis * Dim]; // σ_xy [k][0][1]
        }
    }
}

} // namespace tndm
