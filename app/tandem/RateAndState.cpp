#include "RateAndState.h"
#include "config.h"
#include "kernels/rate_and_state/kernel.h"
#include "kernels/rate_and_state/tensor.h"
#include "tandem/BP1.h"

#include "basis/GaussLegendre.h"
#include "geometry/Vector.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tndm {

RateAndState::RateAndState(Curvilinear<DomainDimension> const& cl,
                           std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints)
    : cl_(&cl), space_(PolynomialDegree - 1, GaussLegendreFactory()) {

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

void RateAndState::prepare(std::size_t faultNo, FacetInfo const& info,
                           std::array<double, DomainDimension> const& ref_normal,
                           LinearAllocator& scratch) {
    auto nbf = space_.numBasisFunctions();
    auto coords =
        Tensor(fault_[faultNo].template get<Coords>().data()->data(), cl_->mapResultInfo(nbf));
    cl_->map(info.up[0], geoE_q[info.localNo[0]], coords);

    double* J_mem = scratch.allocate<double>(nbf * DomainDimension * DomainDimension);
    double* JInv_mem = scratch.allocate<double>(nbf * DomainDimension * DomainDimension);
    double* detJ_mem = scratch.allocate<double>(nbf);
    auto J = Tensor(J_mem, cl_->jacobianResultInfo(nbf));
    auto JInv = Tensor(JInv_mem, cl_->jacobianResultInfo(nbf));
    auto detJ = Tensor(detJ_mem, cl_->detJResultInfo(nbf));
    auto normal = Tensor(fault_[faultNo].template get<UnitNormal>().data()->data(),
                         cl_->normalResultInfo(nbf));
    cl_->jacobian(info.up[0], geoDxi_q[info.localNo[0]], J);
    cl_->detJ(info.up[0], J, detJ);
    cl_->jacobianInv(J, JInv);
    cl_->normal(info.localNo[0], detJ, JInv, normal);
    cl_->normalize(normal);
    for (std::size_t i = 0; i < nbf; ++i) {
        auto& sign_flipped = fault_[faultNo].template get<SignFlipped>()[i];
        auto& normal_i = fault_[faultNo].template get<UnitNormal>()[i];
        sign_flipped = dot(ref_normal, normal_i) < 0;
        if (sign_flipped) {
            normal_i = -1.0 * normal_i;
        }
    }
    scratch.reset();
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

double RateAndState::rhs(std::size_t faultNo, Matrix<double> const& grad_u,
                         Vector<double const>& state, Vector<double>& result,
                         LinearAllocator& scratch) const {
    BP1 bp1;

    std::size_t nbf = space_.numBasisFunctions();
    double* traction = scratch.allocate<double>(nbf);
    rate_and_state::kernel::evaluate_traction krnl;
    krnl.grad_u = grad_u.data();
    krnl.traction = traction;
    krnl.unit_normal = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.execute();

    double VMax = 0.0;
    auto coords = fault_[faultNo].get<Coords>();
    for (std::size_t node = 0; node < nbf; ++node) {
        bp1.setX(coords[node]);
        auto tau = traction[node];
        auto psi = state(node);
        double V = bp1.computeSlipRate(tau, psi);
        assert(std::fabs(bp1.F(tau, V, psi)) < 1e-13);
        VMax = std::max(VMax, std::fabs(V));
        result(node) = bp1.G(tau, V, psi);
        result(nbf + node) = V;
    }
    scratch.reset();
    return VMax;
}

void RateAndState::slip(std::size_t faultNo, Vector<double const>& state, Matrix<double>& s_q,
                        LinearAllocator& scratch) const {
    std::size_t nbf = space_.numBasisFunctions();
    double const* slip = state.data() + nbf;
    double* slip_flip = scratch.allocate<double>(nbf);
    for (std::size_t i = 0; i < nbf; ++i) {
        if (!fault_[faultNo].template get<SignFlipped>()[i]) {
            slip_flip[i] = -slip[i];
        } else {
            slip_flip[i] = slip[i];
        }
    }
    assert(s_q.shape(0) == 1);
    assert(s_q.shape(1) == rate_and_state::tensor::slip_q::size());
    rate_and_state::kernel::evaluate_slip krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.slip = slip_flip;
    krnl.slip_q = s_q.data();
    krnl.execute();
    scratch.reset();
}

void RateAndState::state(std::size_t faultNo, Matrix<double> const& grad_u,
                         Vector<double const>& state, Matrix<double>& result,
                         LinearAllocator& scratch) const {
    BP1 bp1;

    std::size_t nbf = space_.numBasisFunctions();
    double* traction = scratch.allocate<double>(nbf);
    rate_and_state::kernel::evaluate_traction krnl;
    krnl.grad_u = grad_u.data();
    krnl.traction = traction;
    krnl.unit_normal = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.execute();

    auto coords = fault_[faultNo].get<Coords>();
    for (std::size_t node = 0; node < nbf; ++node) {
        bp1.setX(coords[node]);
        auto tau = traction[node];
        auto psi = state(node);
        auto V = bp1.computeSlipRate(tau, psi);
        result(node, 0) = psi;
        result(node, 1) = state(node + nbf);
        result(node, 2) = tau;
        result(node, 3) = V;
    }
    scratch.reset();
}

} // namespace tndm
