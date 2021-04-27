#include "PointLocator.h"

#include "tensor/EigenMap.h"

#include <Eigen/LU>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tndm {

template <std::size_t D>
PointLocator<D>::PointLocator(std::shared_ptr<Curvilinear<D>> cl, double atol, double rtol,
                              double bbox_factor, double max_its)
    : cl_(std::move(cl)), atol_(atol), rtol_(rtol), max_its_(max_its), bbox_(cl_->numElements()) {

    auto E = cl_->evaluateBasisAt(cl_->refElement().refNodes());
    auto num = cl_->refElement().refNodes().size();
    auto F = Managed(cl_->mapResultInfo(num));

    for (std::size_t elNo = 0; elNo < cl_->numElements(); ++elNo) {
        cl_->map(elNo, E, F);
        for (std::size_t d = 0; d < D; ++d) {
            auto& bb = bbox_[elNo][d];
            bb[0] = std::numeric_limits<double>::max();
            bb[1] = std::numeric_limits<double>::lowest();
            for (std::size_t i = 0; i < num; ++i) {
                bb[0] = std::min(bb[0], F(d, i));
                bb[1] = std::max(bb[1], F(d, i));
            }
            double r = (bb[1] - bb[0]) / 2.0;
            double mid = (bb[1] + bb[0]) / 2.0;
            bb[0] = mid - bbox_factor * r;
            bb[1] = mid + bbox_factor * r;
        }
    }
}

template <std::size_t D>
bool PointLocator<D>::inside_bbox(std::size_t elNo, point_t const& point) const {
    bool inside = true;
    auto& bb = bbox_[elNo];
    for (std::size_t d = 0; d < D; ++d) {
        inside = inside && (point[d] >= bb[d][0]) && (point[d] <= bb[d][1]);
    }
    return inside;
}

template <std::size_t D>
auto PointLocator<D>::xInv(std::size_t elNo, point_t const& point, XInvTmp& tmp) const -> point_t {
    point_t xi;
    xi.fill(1.0 / 3.0);
    std::size_t it = 0;

    double residual2_0 = 0.0;
    auto E = cl_->evaluateBasisAt({xi});
    cl_->map(elNo, E, tmp.F);
    auto X_0 = tmp.F.subtensor(slice{}, 0);
    auto F = EigenMap(X_0);
    for (std::size_t d = 0; d < D; ++d) {
        F(d) = point[d] - F(d);
        residual2_0 = F(d) * F(d);
    }

    while (it < max_its_) {
        // point - X(xi)
        auto E = cl_->evaluateBasisAt({xi});
        cl_->map(elNo, E, tmp.F);
        auto X_0 = tmp.F.subtensor(slice{}, 0);
        auto F = EigenMap(X_0);
        double residual2 = 0.0;
        for (std::size_t d = 0; d < D; ++d) {
            F(d) = point[d] - F(d);
            residual2 = F(d) * F(d);
        }
        if (residual2 < atol_ + rtol_ * residual2 / residual2_0) {
            break;
        }

        // JX(xi)
        auto gradE = cl_->evaluateGradientAt({xi});
        cl_->jacobian(elNo, gradE, tmp.J);
        auto J_0 = tmp.J.subtensor(slice{}, slice{}, 0);
        auto J = EigenMap(J_0);

        Eigen::Matrix<double, D, 1> h = J.fullPivLu().solve(F);
        for (std::size_t d = 0; d < D; ++d) {
            xi[d] += h(d);
        }

        ++it;
    }
    if (it == max_its_) {
        throw std::runtime_error("PointLocator: xInv did not converge.");
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
auto BoundaryPointLocator<D>::locate(point_t const& point) const -> BoundaryPointLocatorResult<D> {
    auto r = BoundaryPointLocatorResult<D>{
        std::numeric_limits<std::size_t>::max(), std::numeric_limits<double>::max(), {}, {}};

    auto F = Managed(pl_->cl().mapResultInfo(1));

    auto [closest_elNo, el_dist, closest_xi, dummy] =
        pl_->locate(point, elNos_.begin(), elNos_.end());
    for (auto it = bnds_.find(closest_elNo); it != bnds_.end(); ++it) {
        auto fctNo = it->second.first;
        auto localFaceNo = it->second.second;
        auto [boundary_dist, xi, bary] = closest_[localFaceNo].closest(closest_xi);

        std::array<double, D - 1> chi;
        std::copy(bary.begin() + 1, bary.end(), chi.begin());
        auto E = pl_->cl().evaluateBasisAt({pl_->cl().facetParam(localFaceNo, chi)});
        pl_->cl().map(closest_elNo, E, F);
        point_t x{};
        for (std::size_t d = 0; d < D; ++d) {
            x[d] = F(d, 0);
        }

        double dist = norm(x - point);
        if (dist < r.dist) {
            r.dist = dist;
            r.no = fctNo;
            r.chi = chi;
            r.x = x;
        }
    }

    return r;
} // namespace tndm

template class PointLocator<1ul>;
template class PointLocator<2ul>;
template class PointLocator<3ul>;
template class BoundaryPointLocator<2ul>;
template class BoundaryPointLocator<3ul>;

} // namespace tndm
