#include "PointLocator.h"

#include "tensor/EigenMap.h"

#include <Eigen/LU>
#include <utility>

namespace tndm {

template <std::size_t D>
auto PointLocator<D>::xInv(std::size_t elNo, point_t const& point, XInvTmp& tmp) const -> point_t {
    auto xi = point;
    std::size_t it = 0;
    while (it < max_its_) {
        // point - X(xi)
        auto E = cl_->evaluateBasisAt({xi});
        cl_->map(elNo, E, tmp.F);
        auto X_0 = tmp.F.subtensor(slice{}, 0);
        auto F = EigenMap(X_0);
        double residual = 0.0;
        for (std::size_t d = 0; d < D; ++d) {
            F(d) = point[d] - F(d);
            residual = F(d) * F(d);
        }
        if (residual < tol_) {
            break;
        }

        // JX(xi)
        auto gradE = cl_->evaluateGradientAt({xi});
        cl_->jacobian(elNo, gradE, tmp.J);
        auto J_0 = tmp.J.subtensor(slice{}, slice{}, 0);
        auto J = EigenMap(J_0);

        Eigen::Matrix<double, D, 1> h = J.partialPivLu().solve(F);
        for (std::size_t d = 0; d < D; ++d) {
            xi[d] += h(d);
        }

        ++it;
    }
    return xi;
}

template <std::size_t D>
BoundaryPointLocator<D>::BoundaryPointLocator(std::shared_ptr<PointLocator<D>> pl,
                                              LocalSimplexMesh<D> const& mesh,
                                              std::vector<std::size_t> const& fctNos)
    : pl_(std::move(pl)) {
    for (auto const& fctNo : fctNos) {
        auto elNos = mesh.template upward<D - 1u>(fctNo);
        for (auto elNo : elNos) {
            elNos_.emplace(elNo);
            auto dws = mesh.template downward<D - 1u, D>(elNo);
            int localFaceNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));
            assert(localFaceNo < D + 1u);
            bnds_.emplace(elNo, std::make_pair(fctNo, localFaceNo));
        }
    }

    Simplex<D> refPlex = Simplex<D>::referenceSimplex();
    auto refVertices = Simplex<D>::referenceSimplexVertices();
    auto f2v = refPlex.downward();
    assert(f2v.size() == closest_.size());

    for (std::size_t f = 0; f < f2v.size(); ++f) {
        std::array<std::array<double, D>, D> verts;
        for (std::size_t i = 0; i < D; ++i) {
            verts[i] = refVertices[f2v[f][i]];
        }
        closest_[f] = SimplexDistance(verts);
    }
}

template <std::size_t D>
auto BoundaryPointLocator<D>::locate(point_t const& point) const -> PointLocatorResult<D - 1> {
    auto r = PointLocatorResult<D - 1>{
        std::numeric_limits<std::size_t>::max(), std::numeric_limits<double>::max(), {}};

    auto [closest_elNo, el_dist, closest_xi] = pl_->locate(point, elNos_.begin(), elNos_.end());
    for (auto it = bnds_.find(closest_elNo); it != bnds_.end(); ++it) {
        auto fctNo = it->second.first;
        auto localFaceNo = it->second.second;
        auto [dist, chi, bary] = closest_[localFaceNo].closest(closest_xi);
        if (dist < r.dist) {
            r.dist = dist;
            r.no = fctNo;
            std::copy(bary.begin() + 1, bary.end(), r.xi.begin());
        }
    }

    return r;
}

template class PointLocator<1ul>;
template class PointLocator<2ul>;
template class PointLocator<3ul>;
template class BoundaryPointLocator<2ul>;
template class BoundaryPointLocator<3ul>;

} // namespace tndm
