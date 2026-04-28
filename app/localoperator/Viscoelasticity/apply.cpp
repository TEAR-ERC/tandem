// Matrix-free apply interface 
//
// The adapter methods called by DGOperator.

#include "Viscoelasticity.h"

#include "kernels/viscoelasticity/tensor.h"
#include "util/LinearAllocator.h"
#include <cassert>

namespace tensor = tndm::viscoelasticity::tensor;

namespace tndm {

// Matrix-free apply (wave_rhs) — currently disabled, bodies commented out

template <bool WithRHS>
void Viscoelasticity::apply_(std::size_t elNo, mneme::span<SideInfo> info,
                             Vector<double const> const& x_0,
                             std::array<Vector<double const>, NumFacets> const& x_n,
                             Vector<double>& y_0) const {
    // TODO: implement matrix-free apply with effective VE params (A_dt_q, B_dt_q).
    // The elastic apply_ can be adapted by substituting A_dt/B_dt for lam/mu.
    (void)elNo;
    (void)info;
    (void)x_0;
    (void)x_n;
    (void)y_0;
}

void Viscoelasticity::apply(std::size_t elNo, mneme::span<SideInfo> info,
                            Vector<double const> const& x_0,
                            std::array<Vector<double const>, NumFacets> const& x_n,
                            Vector<double>& y_0) const {
    apply_<false>(elNo, std::move(info), x_0, x_n, y_0);
}

void Viscoelasticity::wave_rhs(std::size_t elNo, mneme::span<SideInfo> info,
                               Vector<double const> const& x_0,
                               std::array<Vector<double const>, NumFacets> const& x_n,
                               Vector<double>& y_0) const {
    // TODO: implement wave_rhs with effective VE params (A_dt_q, B_dt_q).
}

// Output and coefficient extraction

void Viscoelasticity::project(std::size_t elNo, volume_functional_t x, Vector<double>& y) const {
    // TODO: implement projection of effective VE parameters.
}

std::size_t Viscoelasticity::flops_apply(std::size_t elNo, mneme::span<SideInfo> info) const {
    return 0;
}

void Viscoelasticity::coefficients_volume(std::size_t elNo, Matrix<double>& C,
                                          LinearAllocator<double>&) const {
    auto const coeff_lam = material[elNo].get<lam>();
    auto const coeff_mu0 = material[elNo].get<mu0>();
    auto const coeff_mu1 = material[elNo].get<mu1>();
    auto const coeff_viscosity = material[elNo].get<viscosity>();
    assert(coeff_lam.size() == C.shape(0));
    assert(4 == C.shape(1));
    for (std::size_t i = 0; i < C.shape(0); ++i) {
        C(i, 0) = coeff_lam[i];
        C(i, 1) = coeff_mu0[i];
        C(i, 2) = coeff_mu1[i];
        C(i, 3) = coeff_viscosity[i];
    }
}

} // namespace tndm
