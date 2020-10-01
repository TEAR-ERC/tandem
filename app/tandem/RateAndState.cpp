#include "RateAndState.h"
#include "config.h"
#include "tandem/BP1.h"

#include "basis/WarpAndBlend.h"

#include <algorithm>

namespace tndm {

RateAndState::RateAndState(Curvilinear<DomainDimension> const& cl)
    : cl_(&cl), space_(PolynomialDegree, WarpAndBlendFactory<DomainDimension - 1u>()) {

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl.facetParam(f, space_.refNodes());
        geoE_q.emplace_back(cl.evaluateBasisAt(facetParam));
        geoDxi_q.emplace_back(cl.evaluateGradientAt(facetParam));
    }
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

void RateAndState::initial(std::size_t faultNo, Vector<double>& B, LinearAllocator& scratch) const {
    BP1 bp1;

    std::size_t nbf = space_.numBasisFunctions();
    auto coords = fault_[faultNo].get<Coords>();
    for (std::size_t node = 0; node < nbf; ++node) {
        bp1.setX(coords[node]);
        B(node) = bp1.psi0();
        B(nbf + node) = 0.0;
    }
}

void RateAndState::rhs(std::size_t faultNo, Vector<double>& B, LinearAllocator& scratch) const {
    /*double tractionBuf[tensor::grad_u::size()];
    auto traction = Matrix<double>(tractionBuf, tensor::grad_u::Shape[0], tensor::grad_u::Shape[1]);
    for (std::size_t faultNo = 0; faultNo < info_.size(); ++faultNo) {
        auto coords = info_[faultNo].get<Coords>();
        poisson.grad_u(fctNos_[faultNo], U, traction);
        auto tau = info_[faultNo].get<Tau>();
        auto V = info_[faultNo].get<SlipRate>();
        for (std::size_t node = 0; node < nbf; ++node) {
            bp1.setX(coords[node]);
            tau[node] = traction(0, node);
            if (coords[node][1] <= -40000.0) {
                V[node] = 1e-9;
                F(node, 0, faultNo) = 0.0;
                F(node, 1, faultNo) = V[node];
            } else {
                double psi = X(node, 0, faultNo);
                double Vn = bp1.computeSlipRate(tau[node], psi);
                F(node, 0, faultNo) = bp1.G(tau[node], Vn, psi);
                F(node, 1, faultNo) = Vn;
                V[node] = Vn;
            }
            VMax_ = std::max(VMax_, V[node]);
        }*/
}

} // namespace tndm
