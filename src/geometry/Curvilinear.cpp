#include "Curvilinear.h"
#include "Affine.h"
#include "Vector.h"
#include "basis/Equidistant.h"
#include "basis/Util.h"
#include "mesh/LocalSimplexMesh.h"
#include "mesh/MeshData.h"
#include "tensor/EigenMap.h"
#include "tensor/Managed.h"
#include "tensor/Reshape.h"
#include "util/Math.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace tndm {

template <std::size_t D>
Curvilinear<D>::Curvilinear(LocalSimplexMesh<D> const& mesh, transform_t transform, unsigned degree,
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
    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh.facets().data());
    if (boundaryData) {
        facetTags = boundaryData->getFacetTags();
    } else {
        std::cerr << "Warning: Facet tags are not set in the mesh. Setting to a default of -1"
                  << std::endl;
        facetTags.resize(mesh.numElements(), -1);
    }
    auto volumeData = dynamic_cast<VolumeData const*>(mesh.elements().getVolumeData());
    if (volumeData) {
        volumeTags = volumeData->getVolumeTags();
    } else {
        std::cerr << "Warning: Volume tags are not set in the mesh. Setting to a default of -1"
                  << std::endl;
        volumeTags.resize(mesh.numElements(), -1);
    }
    auto elementData = dynamic_cast<ElementData const*>(mesh.elements().data());
    Managed<Matrix<double>> eval_basis;
    if (elementData) {
        auto const& nodes = elementData->getNodes();
        // Vertices are excluded in nodes, therefore add D + 1
        std::size_t num_bf = nodes.shape(1) + D + 1u;
        auto N = num_nodes_to_degree<D>(num_bf);
        if (!N) {
            std::stringstream s;
            s << "The number of nodes " << num_bf << " is incomplete.";
            throw std::runtime_error(s.str());
        }
        auto espace = NodalRefElement<D>(
            *N, EquidistantNodesFactory<D>(elementData->getNumberingConvention()));
        eval_basis = espace.evaluateBasisAt(refElement_.refNodes());
    }

    std::size_t vertexNo = 0;
    std::size_t elNo = 0;
    local_mesh_size_ = 0.0;
    for (auto const& element : mesh.elements()) {
        auto vlids = mesh.template downward<0>(element);
        std::array<std::array<double, D>, D + 1> verts;
        std::size_t localVertexNo = 0;
        for (auto const& vlid : vlids) {
            verts[localVertexNo++] = vertexData->getVertices()[vlid];
        }

        if (elementData) {
            constexpr std::size_t NumVerts = D + 1u;
            auto const& nodes = elementData->getNodes();
            for (std::size_t j = 0; j < eval_basis.shape(1); ++j) {
                auto vtx = std::array<double, D>{};
                std::size_t i = 0;
                for (; i < NumVerts; ++i) {
                    for (std::size_t d = 0; d < D; ++d) {
                        vtx[d] += verts[i][d] * eval_basis(i, j);
                    }
                }
                for (; i < eval_basis.shape(0); ++i) {
                    for (std::size_t d = 0; d < D; ++d) {
                        vtx[d] += nodes(d, i - NumVerts, elNo) * eval_basis(i, j);
                    }
                }
                (*storage)[vertexNo] = transform(vtx);
                ++vertexNo;
            }
        } else {
            RefPlexToGeneralPlex<D> map(verts);
            for (auto& refNode : refElement_.refNodes()) {
                (*storage)[vertexNo] = transform(map(refNode));
                ++vertexNo;
            }
        }
        ++elNo;

        for (auto& x : verts) {
            x = transform(x);
        }
        for (auto const& x : verts) {
            for (auto const& y : verts) {
                auto h = norm(x - y);
                local_mesh_size_ = std::max(local_mesh_size_, h);
            }
        }
    }

    Simplex<D> refPlex = Simplex<D>::referenceSimplex();
    f2v = refPlex.downward();

    // Compute reference normals
    std::size_t fsNo = 0;
    for (auto const& f : f2v) {
        std::array<double, D> normal;
        if constexpr (D == 1u) {
            normal[0] = 1.0;
        } else if constexpr (D == 2u) {
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
TensorBase<Vector<long int>> Curvilinear<D>::tagsInfo(std::size_t numPoints) const {
    return TensorBase<Vector<long int>>(numPoints);
}

template <std::size_t D>
void Curvilinear<D>::setVolumeTags(std::size_t eleNo, Tensor<long int, 1u>& result) const {
    for (std::ptrdiff_t i = 0; i < result.shape(0); ++i) {
        result(i) = volumeTags[eleNo];
    }
}

template <std::size_t D>
void Curvilinear<D>::setFacetTags(FacetInfo const& info, Tensor<long int, 1u>& result) const {
    for (std::ptrdiff_t i = 0; i < result.shape(0); ++i) {
        result(i) = info.facetTag;
    }
}

template <std::size_t D>
void Curvilinear<D>::map(std::size_t eleNo, Matrix<double> const& E,
                         Tensor<double, 2u>& result) const {
    assert(eleNo < vertices.size());
    assert(result.shape(0) == D);
    assert(result.shape(1) == E.shape(1));
    assert(E.shape(0) == refElement_.numBasisFunctions());

    auto vertexSpan = vertices[eleNo];
    assert(vertexSpan.size() == refElement_.numBasisFunctions());
    Eigen::Map<Eigen::Matrix<double, D, Eigen::Dynamic>> vertMap(vertexSpan.data()->data(), D,
                                                                 refElement_.numBasisFunctions());
    EigenMap(result) = vertMap * EigenMap(E);
}

template <std::size_t D>
TensorBase<Tensor<double, 3u>> Curvilinear<D>::jacobianResultInfo(std::size_t numPoints) const {
    return TensorBase<Tensor<double, 3u>>(D, D, numPoints);
}

template <std::size_t D>
void Curvilinear<D>::jacobian(std::size_t eleNo, Tensor<double, 3u> const& gradE,
                              Tensor<double, 3u>& result) const {
    assert(eleNo < vertices.size());

    auto vertexSpan = vertices[eleNo];
    assert(vertexSpan.size() == refElement_.numBasisFunctions());
    Eigen::Map<Eigen::Matrix<double, D, Eigen::Dynamic>> vertMap(vertexSpan.data()->data(), D,
                                                                 refElement_.numBasisFunctions());

    assert(gradE.shape(0) == refElement_.numBasisFunctions());
    assert(gradE.shape(1) == D);
    auto gradEMat = reshape(gradE, refElement_.numBasisFunctions(), D * gradE.shape(2));

    assert(result.shape(0) == D);
    assert(result.shape(1) == D);
    assert(result.shape(2) == gradE.shape(2));
    auto resultMat = reshape(result, D, D * gradE.shape(2));

    EigenMap(resultMat) = vertMap * EigenMap(gradEMat);
}

template <std::size_t D>
void Curvilinear<D>::jacobianInv(Tensor<double, 3u> const& jacobian,
                                 Tensor<double, 3u>& result) const {
    for (std::ptrdiff_t i = 0; i < result.shape(2); ++i) {
        auto jAtP = jacobian.subtensor(slice{}, slice{}, i);
        auto resAtP = result.subtensor(slice{}, slice{}, i);
        EigenMap<Matrix<double>, D, D>(resAtP) =
            EigenMap<const Matrix<double>, D, D>(jAtP).inverse();
    }
}

template <std::size_t D>
TensorBase<Vector<double>> Curvilinear<D>::detJResultInfo(std::size_t numPoints) const {
    return TensorBase<Vector<double>>(numPoints);
}

template <std::size_t D>
void Curvilinear<D>::detJ(std::size_t eleNo, Tensor<double, 3u> const& jacobian,
                          Tensor<double, 1u>& result) const {
    for (std::ptrdiff_t i = 0; i < result.shape(0); ++i) {
        auto jAtP = jacobian.subtensor(slice{}, slice{}, i);
        result(i) = EigenMap<const Matrix<double>, D, D>(jAtP).determinant();
    }
}

template <std::size_t D>
void Curvilinear<D>::absDetJ(std::size_t eleNo, Tensor<double, 3u> const& jacobian,
                             Tensor<double, 1u>& result) const {
    for (std::ptrdiff_t i = 0; i < result.shape(0); ++i) {
        auto jAtP = jacobian.subtensor(slice{}, slice{}, i);
        result(i) = std::fabs(EigenMap<const Matrix<double>, D, D>(jAtP).determinant());
    }
}

template <std::size_t D>
TensorBase<Matrix<double>> Curvilinear<D>::normalResultInfo(std::size_t numPoints) const {
    return TensorBase<Matrix<double>>(D, numPoints);
}

template <std::size_t D>
void Curvilinear<D>::normal(std::size_t faceNo, Tensor<double, 1u> const& detJ,
                            Tensor<double, 3u> const& jInv, Tensor<double, 2u>& result) const {
    assert(faceNo < D + 1u);
    // n_{iq} = |J|_q J^{-T}_{ijq} N_j
    for (std::ptrdiff_t i = 0; i < detJ.shape(0); ++i) {
        auto jInvAtP = jInv.subtensor(slice{}, slice{}, i);
        auto res = result.subtensor(slice{}, i);
        EigenMap<Vector<double>, D>(res) =
            std::fabs(detJ(i)) * EigenMap<const Matrix<double>, D, D>(jInvAtP).transpose() *
            refNormals[faceNo];
    }
}

template <std::size_t D> void Curvilinear<D>::normalize(Tensor<double, 2u>& normal) const {
    for (std::ptrdiff_t i = 0; i < normal.shape(1); ++i) {
        auto n = normal.subtensor(slice{}, i);
        EigenMap<Vector<double>, D>(n).normalize();
    }
}

template <std::size_t D>
TensorBase<Tensor<double, 3u>> Curvilinear<D>::facetBasisResultInfo(std::size_t numPoints) const {
    return TensorBase<Tensor<double, 3u>>(D, D, numPoints);
}

template <std::size_t D>
void Curvilinear<D>::facetBasis(std::array<double, D> const& up, Matrix<double> const& normal,
                                Tensor<double, 3u>& result) const {
    assert(result.shape(2) == normal.shape(1));

    constexpr double colinear_tol = 10000.0 * std::numeric_limits<double>::epsilon();

    for (std::ptrdiff_t i = 0; i < result.shape(2); ++i) {
        auto n_in = normal.subtensor(slice{}, i);
        auto n_in_eigen = EigenMap<Vector<double>, D>(n_in);
        auto n = result.subtensor(slice{}, 0, i);
        auto n_eigen = EigenMap<Vector<double>, D>(n);
        n_eigen = n_in_eigen.normalized();

        if constexpr (D == 2u) {
            double s = sgn(up[0] * n(1) - up[1] * n(0));
            if (std::fabs(s) < colinear_tol) {
                throw std::logic_error("Up vector and normal are almost colinear.");
            }
            auto d = result.subtensor(slice{}, 1, i);
            d(0) = -s * n(1);
            d(1) = s * n(0);
        } else if constexpr (D == 3u) {
            auto u = Eigen::Vector3d(up.data());
            auto d = result.subtensor(slice{}, 1, i);
            auto d_eigen = EigenMap<Vector<double>, D>(d);
            auto s = result.subtensor(slice{}, 2, i);
            auto s_eigen = EigenMap<Vector<double>, D>(s);
            s_eigen = u.cross(n_eigen).normalized();
            if (s_eigen.norm() < colinear_tol) {
                throw std::logic_error("Up vector and normal are almost colinear.");
            }
            d_eigen = s_eigen.cross(n_eigen).normalized();
        }
    }
}

template <std::size_t D>
void Curvilinear<D>::facetBasisFromPlexTangents(std::size_t faceNo,
                                                Tensor<double, 3u> const& jacobian,
                                                Matrix<double> const& normal,
                                                Tensor<double, 3u>& result) const {
    assert(result.shape(2) == normal.shape(1));
    assert(result.shape(2) == jacobian.shape(2));

    auto& f = f2v[faceNo];
    auto refTangent = refVertices[f[1]] - refVertices[f[0]];
    for (std::ptrdiff_t i = 0; i < result.shape(2); ++i) {
        auto n = normal.subtensor(slice{}, i);
        auto n_eigen = EigenMap<Vector<double>, D>(n);
        auto n_res = result.subtensor(slice{}, 0, i);
        auto n_res_eigen = EigenMap<Vector<double>, D>(n_res);
        n_res_eigen = n_eigen.normalized();

        if constexpr (D >= 2u) {
            auto jAtP = jacobian.subtensor(slice{}, slice{}, i);
            auto t1_res = result.subtensor(slice{}, 1, i);
            auto t1_res_eigen = EigenMap<Vector<double>, D>(t1_res);
            // first tangent = J * refTangent
            t1_res_eigen = EigenMap<const Matrix<double>, D, D>(jAtP) *
                           Eigen::Map<Eigen::Matrix<double, D, 1>>(refTangent.data());
            t1_res_eigen.normalize();

            if constexpr (D == 3u) {
                auto t2_res = result.subtensor(slice{}, 2, i);
                auto t2_res_eigen = EigenMap<Vector<double>, D>(t2_res);
                t2_res_eigen = n_res_eigen.cross(t1_res_eigen);
                t2_res_eigen.normalize();
            }
        }
    }
}

template <std::size_t D>
std::array<double, D> Curvilinear<D>::facetParam(std::size_t faceNo,
                                                 std::array<double, D - 1> const& chi) const {
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
Curvilinear<D>::facetParam(std::size_t faceNo,
                           std::vector<std::array<double, D - 1>> const& chis) const {
    std::vector<std::array<double, D>> xis;
    xis.reserve(chis.size());
    for (auto const& chi : chis) {
        xis.emplace_back(facetParam(faceNo, chi));
    }
    return xis;
}

template class Curvilinear<1ul>;
template class Curvilinear<2ul>;
template class Curvilinear<3ul>;

} // namespace tndm
