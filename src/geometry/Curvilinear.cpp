#include "Curvilinear.h"
#include "Affine.h"
#include "Vector.h"
#include "basis/Functions.h"
#include "util/Combinatorics.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <stdexcept>

namespace tndm {

template <std::size_t D>
Curvilinear<D>::Curvilinear(LocalSimplexMesh<D> const& mesh,
                            std::function<vertex_t(vertex_t const&)> transform, unsigned degree,
                            NodesFactory<D> const& nodesFactory)
    : N(degree) {
    // Get vertices
    std::vector<std::array<double, D>> refNodes = nodesFactory(degree);
    std::size_t vertsPerElement = refNodes.size();
    assert(binom(N + D, D) == refNodes.size());

    storage.resize(mesh.numElements() * vertsPerElement);
    vertices.setStorage(storage, 0, mesh.numElements(), vertsPerElement);

    auto vertexData = dynamic_cast<VertexData<D> const*>(mesh.vertices().data());
    if (!vertexData) {
        throw std::runtime_error("Expected vertex data");
    }

    std::size_t vertexNo = 0;
    for (auto const& element : mesh.elements()) {
        auto vlids = mesh.template downward<0>(element);
        std::array<std::array<double, D>, D + 1> verts;
        std::size_t localVertexNo = 0;
        for (auto const& vlid : vlids) {
            verts[localVertexNo++] = vertexData->getVertices()[vlid];
        }
        RefPlexToGeneralPlex<D> map(verts);
        for (auto& refNode : refNodes) {
            storage[vertexNo] = transform(map(refNode));
            ++vertexNo;
        }
    }

    // Compute Vandermonde matrix
    Simplex<D> refPlex = Simplex<D>::referenceSimplex();
    f2v = refPlex.downward();

    vandermonde = Vandermonde(N, refNodes);
    vandermondeInvT = vandermonde.inverse().transpose();

    // Compute reference normals
    std::size_t fsNo = 0;
    for (auto const& f : f2v) {
        std::array<double, D> normal;
        if constexpr (D == 2u) {
            // Compute normal by rotating edge
            normal = refVertices[f[1]] - refVertices[f[0]];
            std::swap(normal[0], normal[1]);
            normal[0] *= -1.0;
        } else {
            // Compute normal by cross product
            auto e1 = refVertices[f[1]] - refVertices[f[0]];
            auto e2 = refVertices[f[2]] - refVertices[f[0]];
            normal = cross(e1, e2);
        }

        refNormals[fsNo] = Eigen::Map<Eigen::Matrix<double, D, 1>>(normal.data());

        std::vector<uint64_t> missingVertex;
        std::set_difference(refPlex.begin(), refPlex.end(), f.begin(), f.end(),
                            std::inserter(missingVertex, missingVertex.begin()));
        assert(missingVertex.size() == 1);

        auto edge = refVertices[missingVertex[0]] - refVertices[f[0]];
        if (dot(normal, edge) > 0) {
            refNormals[fsNo] *= -1.0;
        }
        ++fsNo;
    }
}

template <std::size_t D>
std::array<double, D> Curvilinear<D>::map(std::size_t eleNo, std::array<double, D> const& xi) {
    assert(eleNo < vertices.size());

    Eigen::VectorXd phi(vandermondeInvT.cols());
    std::size_t bf = 0;
    for (auto j : AllIntegerSums<D>(N)) {
        phi(bf++) = DubinerP(j, xi);
    }
    auto vertexSpan = vertices[eleNo];
    assert(vertexSpan.size() == vandermondeInvT.rows());

    std::array<double, D> result;
    Eigen::Map<Eigen::Matrix<double, D, Eigen::Dynamic>> vertMap(vertexSpan.data()->data(), D,
                                                                 vandermondeInvT.rows());
    Eigen::Map<Eigen::Matrix<double, D, 1>>(result.data()) = vertMap * (vandermondeInvT * phi);
    return result;
}

template <std::size_t D>
std::array<double, D * D> Curvilinear<D>::jacobian(std::size_t eleNo,
                                                   std::array<double, D> const& xi) {
    assert(eleNo < vertices.size());

    Eigen::MatrixXd phi(vandermondeInvT.cols(), D);
    std::size_t bf = 0;
    for (auto j : AllIntegerSums<D>(N)) {
        auto dphi = gradDubinerP(j, xi);
        for (std::size_t d = 0; d < D; ++d) {
            phi(bf, d) = dphi[d];
        }
        ++bf;
    }
    auto vertexSpan = vertices[eleNo];
    assert(vertexSpan.size() == vandermondeInvT.rows());

    std::array<double, D * D> result;
    Eigen::Map<Eigen::Matrix<double, D, Eigen::Dynamic>> vertMap(vertexSpan.data()->data(), D,
                                                                 vandermondeInvT.rows());
    Eigen::Map<Eigen::Matrix<double, D, D>>(result.data()) = vertMap * (vandermondeInvT * phi);
    return result;
}

template <std::size_t D>
double Curvilinear<D>::detJ(std::size_t eleNo, std::array<double, D> const& xi) {
    auto J = jacobian(eleNo, xi);
    return Eigen::Map<Eigen::Matrix<double, D, D>>(J.data()).determinant();
}

template <std::size_t D>
std::array<double, D> Curvilinear<D>::normal(std::size_t eleNo, std::size_t faceNo,
                                             std::array<double, D> const& xi) {
    auto Jraw = jacobian(eleNo, xi);
    auto J = Eigen::Map<Eigen::Matrix<double, D, D>>(Jraw.data());
    std::array<double, D> result;
    Eigen::Map<Eigen::Matrix<double, D, 1>>(result.data()) =
        std::fabs(J.determinant()) * J.transpose().colPivHouseholderQr().solve(refNormals[faceNo]);
    return result;
}

template <std::size_t D>
std::array<double, D> Curvilinear<D>::facetParam(std::size_t faceNo,
                                                 std::array<double, D - 1> const& chi) {
    auto& f = f2v[faceNo];
    std::array<double, D> xi;
    double chiSum = 0.0;
    for (std::size_t d = 0; d < D - 1; ++d) {
        chiSum += chi[d];
    }
    xi = (1.0 - chiSum) * refVertices[f[0]];
    for (std::size_t d = 0; d < D - 1; ++d) {
        xi = xi + chi[d] * refVertices[f[d + 1]];
    }
    return xi;
}

template class Curvilinear<2u>;
template class Curvilinear<3u>;

} // namespace tndm
