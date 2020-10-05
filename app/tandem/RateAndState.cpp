#include "RateAndState.h"
#include "config.h"
#include "kernels/rate_and_state/kernel.h"
#include "tandem/BP1.h"

#include "basis/WarpAndBlend.h"

#include <algorithm>

namespace tndm {

RateAndState::RateAndState(Curvilinear<DomainDimension> const& cl,
                           std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints)
    : cl_(&cl), space_(PolynomialDegree, WarpAndBlendFactory<DomainDimension - 1u>()) {

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl.facetParam(f, space_.refNodes());
        geoE_q.emplace_back(cl.evaluateBasisAt(facetParam));
        geoDxi_q.emplace_back(cl.evaluateGradientAt(facetParam));
    }

    e_q_T = space_.evaluateBasisAt(quadPoints, {1, 0});
}

void RateAndState::begin_preparation(std::size_t numFaultFaces) {
    auto nbf = space_.numBasisFunctions();
    fault_.setStorage(std::make_shared<fault_t>(numFaultFaces * nbf), 0u, numFaultFaces, nbf);
}

void RateAndState::prepare(std::size_t faultNo, FacetInfo const& info, LinearAllocator& scratch) {
    auto nbf = space_.numBasisFunctions();
    auto coords =
        Tensor(fault_[faultNo].template get<Coords>().data()->data(), cl_->mapResultInfo(nbf));
    cl_->map(info.up[0], geoE_q[info.localNo[0]], coords);
}

void RateAndState::initial(std::size_t faultNo, Vector<double>& state,
                           LinearAllocator& scratch) const {
    BP1 bp1;

    std::size_t nbf = space_.numBasisFunctions();
    auto coords = fault_[faultNo].get<Coords>();
    for (std::size_t node = 0; node < nbf; ++node) {
        bp1.setX(coords[node]);
        state(node) = bp1.psi0();
        state(nbf + node) = 0.0;
    }
}

void RateAndState::rhs(std::size_t faultNo, Matrix<double> const& traction,
                       Vector<double const>& state, Vector<double>& result,
                       LinearAllocator& scratch) const {
    BP1 bp1;

    std::size_t nbf = space_.numBasisFunctions();
    auto coords = fault_[faultNo].get<Coords>();
    for (std::size_t node = 0; node < nbf; ++node) {
        bp1.setX(coords[node]);
        if (coords[node][1] <= -40000.0) {
            result(node) = 0.0;
            result(nbf + node) = 1e-9;
        } else {
            auto tau = traction(0, node);
            auto psi = state(node);
            double V = bp1.computeSlipRate(tau, psi);
            result(node) = bp1.G(tau, V, psi);
            result(nbf + node) = V;
        }
    }
}

void RateAndState::slip(std::size_t faultNo, Vector<double const>& state,
                        Matrix<double>& s_q) const {
    std::size_t nbf = space_.numBasisFunctions();
    rate_and_state::kernel::evaluate_slip krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.slip = state.data() + nbf;
    krnl.slip_q = s_q.data();
    krnl.execute();
}

} // namespace tndm
