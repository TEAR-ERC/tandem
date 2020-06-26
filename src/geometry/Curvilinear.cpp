#include "Curvilinear.h"
#include "Affine.h"
#include "Vector.h"
#include "form/RefElement.h"
#include "tensor/EigenMap.h"
#include "tensor/Reshape.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace tndm {

template <std::size_t D>
Curvilinear<D>::Curvilinear(LocalSimplexMesh<D> const& mesh,
                            std::function<vertex_t(vertex_t const&)> transform, unsigned degree,
                            NodesFactory<D> const& nodesFactory)
    : N(degree), refElement_(degree, nodesFactory) {
    // Get vertices
    std::size_t vertsPerElement = refElement_.refNodes().size();

    auto storage =
        std::make_shared<mneme::SingleStorage<Verts>>(mesh.numElements() * vertsPerElement);
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
        for (auto& refNode : refElement_.refNodes()) {
            (*storage)[vertexNo] = transform(map(refNode));
            ++vertexNo;
        }
    }

    Simplex<D> refPlex = Simplex<D>::referenceSimplex();
    f2v = refPlex.downward();

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
TensorBase<Matrix<double>> Curvilinear<D>::mapResultInfo(std::size_t numPoints) const {
    return TensorBase<Matrix<double>>(D, numPoints);
}

template <std::size_t D>
void Curvilinear<D>::map(std::size_t eleNo, Matrix<double> const& E, Tensor<double, 2u>& result) {
    assert(eleNo < vertices.size());
    assert(result.shape(0) == D);
    assert(result.shape(1) == E.shape(1));
    assert(E.shape(0) == refElement_.numberOfBasisFunctions());

    auto vertexSpan = vertices[eleNo];
    assert(vertexSpan.size() == refElement_.numberOfBasisFunctions());
    Eigen::Map<Eigen::Matrix<double, D, Eigen::Dynamic>> vertMap(
        vertexSpan.data()->data(), D, refElement_.numberOfBasisFunctions());
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
    assert(vertexSpan.size() == refElement_.numberOfBasisFunctions());
    Eigen::Map<Eigen::Matrix<double, D, Eigen::Dynamic>> vertMap(
        vertexSpan.data()->data(), D, refElement_.numberOfBasisFunctions());

    assert(gradE.shape(0) == refElement_.numberOfBasisFunctions());
    assert(gradE.shape(1) == D);
    auto gradEMat = reshape(gradE, refElement_.numberOfBasisFunctions(), D * gradE.shape(2));

    assert(result.shape(0) == D);
    assert(result.shape(1) == D);
    assert(result.shape(2) == gradE.shape(2));
    auto resultMat = reshape(result, D, D * gradE.shape(2));

    EigenMap(resultMat) = vertMap * EigenMap(gradEMat);
}

template <std::size_t D>
void Curvilinear<D>::jacobianInv(Tensor<double, 3u> const& jacobian, Tensor<double, 3u>& result) {
    for (std::ptrdiff_t i = 0; i < result.shape(2); ++i) {
        auto jAtP = jacobian.subtensor(slice{}, slice{}, i);
        auto resAtP = result.subtensor(slice{}, slice{}, i);
        EigenMap<Matrix<double>, D, D>(resAtP) =
            EigenMap<Matrix<const double>, D, D>(jAtP).inverse();
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
        auto jAtP = jacobian.subtensor(slice{}, slice{}, i);
        result(i) = EigenMap<Matrix<const double>, D, D>(jAtP).determinant();
    }
}

template <std::size_t D>
void Curvilinear<D>::absDetJ(std::size_t eleNo, Tensor<double, 3u> const& jacobian,
                             Tensor<double, 1u>& result) {
    for (std::ptrdiff_t i = 0; i < result.shape(0); ++i) {
        auto jAtP = jacobian.subtensor(slice{}, slice{}, i);
        result(i) = std::fabs(EigenMap<Matrix<const double>, D, D>(jAtP).determinant());
    }
}

template <std::size_t D>
TensorBase<Matrix<double>> Curvilinear<D>::normalResultInfo(std::size_t numPoints) const {
    return TensorBase<Matrix<double>>(D, numPoints);
}

template <std::size_t D>
void Curvilinear<D>::normal(std::size_t faceNo, Tensor<double, 1u> const& detJ,
                            Tensor<double, 3u> const& jInv, Tensor<double, 2u>& result) {
    // n_{iq} = |J|_q J^{-T}_{ijq} N_j
    for (std::ptrdiff_t i = 0; i < detJ.shape(0); ++i) {
        auto jInvAtP = jInv.subtensor(slice{}, slice{}, i);
        auto res = result.subtensor(slice{}, i);
        EigenMap<Vector<double>, D>(res) =
            std::fabs(detJ(i)) * EigenMap<Matrix<const double>, D, D>(jInvAtP).transpose() *
            refNormals[faceNo];
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

template <std::size_t D>
std::vector<std::array<double, D>>
Curvilinear<D>::facetParam(std::size_t faceNo, std::vector<std::array<double, D - 1>> const& chis) {
    std::vector<std::array<double, D>> xis;
    xis.reserve(chis.size());
    for (auto const& chi : chis) {
        xis.emplace_back(facetParam(faceNo, chi));
    }
    return xis;
}

template class Curvilinear<2ul>;
template class Curvilinear<3ul>;

} // namespace tndm
