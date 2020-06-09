#include "Curvilinear.h"
#include "Affine.h"
#include "Vector.h"
#include "basis/Functions.h"
#include "tensor/EigenMap.h"
#include "tensor/Reshape.h"
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
Managed<Matrix<double>>
Curvilinear<D>::evaluateBasisAt(std::vector<std::array<double, D>> const& points) {
    Managed<Matrix<double>> E(vandermondeInvT.cols(), points.size());
    for (std::size_t p = 0; p < points.size(); ++p) {
        std::size_t bf = 0;
        for (auto j : AllIntegerSums<D>(N)) {
            E(bf++, p) = DubinerP(j, points[p]);
        }
    }
    auto Emap = EigenMap(E);
    Emap = vandermondeInvT * Emap;
    return E;
}

template <std::size_t D>
Managed<Tensor<double, 3u>>
Curvilinear<D>::evaluateGradientAt(std::vector<std::array<double, D>> const& points) {
    Managed<Tensor<double, 3u>> gradE(vandermondeInvT.cols(), D, points.size());
    for (std::size_t p = 0; p < points.size(); ++p) {
        std::size_t bf = 0;
        for (auto j : AllIntegerSums<D>(N)) {
            auto dphi = gradDubinerP(j, points[p]);
            for (std::size_t d = 0; d < D; ++d) {
                gradE(bf, d, p) = dphi[d];
            }
            ++bf;
        }
    }

    assert(vandermondeInvT.cols() == vandermondeInvT.rows());
    auto matView = reshape(gradE, vandermondeInvT.cols(), D * points.size());
    auto map = EigenMap(matView);
    map = vandermondeInvT * map;
    return gradE;
}

template <std::size_t D>
TensorBase<Matrix<double>> Curvilinear<D>::mapResultInfo(std::size_t numPoints) const {
    return TensorBase<Matrix<double>>(D, numPoints);
}

template <std::size_t D>
void Curvilinear<D>::map(std::size_t eleNo, Matrix<double> const& E, Tensor<double, 2u>& result) {
    assert(eleNo < vertices.size());
    assert(E.shape(1) == result.shape(1));

    auto vertexSpan = vertices[eleNo];
    assert(vertexSpan.size() == vandermondeInvT.rows());
    Eigen::Map<Eigen::Matrix<double, D, Eigen::Dynamic>> vertMap(vertexSpan.data()->data(), D,
                                                                 vandermondeInvT.rows());
    EigenMap(result) = vertMap * EigenMap(E);
}

template <std::size_t D>
TensorBase<Tensor<double, 3u>> Curvilinear<D>::jacobianResultInfo(std::size_t numPoints) const {
    return TensorBase<Tensor<double, 3u>>(D, D, numPoints);
}

template <std::size_t D>
void Curvilinear<D>::jacobian(std::size_t eleNo, Tensor<double, 3u> const& gradE,
                              Tensor<double, 3u>& result) {
    assert(eleNo < vertices.size());

    auto vertexSpan = vertices[eleNo];
    assert(vertexSpan.size() == vandermondeInvT.rows());
    Eigen::Map<Eigen::Matrix<double, D, Eigen::Dynamic>> vertMap(vertexSpan.data()->data(), D,
                                                                 vandermondeInvT.rows());

    auto gradEMat = reshape(gradE, vandermondeInvT.rows(), D * gradE.shape(2));
    auto resultMat = reshape(result, D, D * gradE.shape(2));
    EigenMap(resultMat) = vertMap * EigenMap(gradEMat);
}

template <std::size_t D>
void Curvilinear<D>::jacobianInvT(Tensor<double, 3u> const& jacobian, Tensor<double, 3u>& result) {
    for (std::ptrdiff_t i = 0; i < result.shape(2); ++i) {
        Eigen::Map<const Eigen::Matrix<double, D, D>> Jmap(&jacobian(0, 0, i));
        Eigen::Map<Eigen::Matrix<double, D, D>> resultMap(&result(0, 0, i));
        resultMap = Jmap.transpose().inverse();
    }
}

template <std::size_t D>
TensorBase<Vector<double>> Curvilinear<D>::detJResultInfo(std::size_t numPoints) const {
    return TensorBase<Vector<double>>(numPoints);
}

template <std::size_t D>
void Curvilinear<D>::detJ(std::size_t eleNo, Tensor<double, 3u> const& jacobian,
                          Tensor<double, 1u>& result) {
    for (std::ptrdiff_t i = 0; i < result.shape(0); ++i) {
        result(i) = Eigen::Map<const Eigen::Matrix<double, D, D>>(&jacobian(0, 0, i)).determinant();
    }
}

template <std::size_t D>
TensorBase<Matrix<double>> Curvilinear<D>::normalResultInfo(std::size_t numPoints) const {
    return TensorBase<Matrix<double>>(D, numPoints);
}

template <std::size_t D>
void Curvilinear<D>::normal(std::size_t faceNo, Tensor<double, 1u> const& detJ,
                            Tensor<double, 3u> const& JinvT, Tensor<double, 2u>& result) {
    for (std::ptrdiff_t i = 0; i < detJ.shape(0); ++i) {
        Eigen::Map<const Eigen::Matrix<double, D, D>> JinvTmap(&JinvT(0, 0, i));
        Eigen::Map<Eigen::Matrix<double, D, 1>> resultMap(&result(0, i));
        resultMap = std::fabs(detJ(i)) * JinvTmap * refNormals[faceNo];
    }
}

template <std::size_t D>
std::array<double, D> Curvilinear<D>::facetParam(std::size_t faceNo,
                                                 std::array<double, D - 1> const& chi) {
    auto& f = f2v[faceNo];
    std::array<double, D> xi;
    double chiSum = 0.0;
    for (std::size_t d = 0; d < chi.size(); ++d) {
        chiSum += chi[d];
    }
    xi = (1.0 - chiSum) * refVertices[f[0]];
    for (std::size_t d = 0; d < chi.size(); ++d) {
        xi = xi + chi[d] * refVertices[f[d + 1]];
    }
    return xi;
}

template class Curvilinear<2u>;
template class Curvilinear<3u>;

} // namespace tndm
