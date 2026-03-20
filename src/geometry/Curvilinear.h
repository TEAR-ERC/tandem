#ifndef CURVILINEAR_20200609_H
#define CURVILINEAR_20200609_H

#include "basis/WarpAndBlend.h"
#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "mesh/Simplex.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"

#include "mneme/storage.hpp"
#include "mneme/view.hpp"
#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace tndm {

template <std::size_t D> class LocalSimplexMesh;

template <std::size_t D> class Curvilinear {
public:
    using vertex_t = std::array<double, D>;
    using transform_t = std::function<vertex_t(vertex_t const&)>;

    Curvilinear(
        LocalSimplexMesh<D> const& mesh,
        transform_t transform = [](vertex_t const& v) { return v; }, unsigned degree = 1,
        NodesFactory<D> const& nodesFactory = WarpAndBlendFactory<D>());

    auto evaluateBasisAt(std::vector<std::array<double, D>> const& points) const {
        return refElement_.evaluateBasisAt(points);
    }
    auto evaluateGradientAt(std::vector<std::array<double, D>> const& points) const {
        return refElement_.evaluateGradientAt(points);
    }

    TensorBase<Matrix<double>> mapResultInfo(std::size_t numPoints) const;
    void map(std::size_t eleNo, Matrix<double> const& E, Tensor<double, 2u>& result) const;

    TensorBase<Tensor<double, 3u>> jacobianResultInfo(std::size_t numPoints) const;
    void jacobian(std::size_t eleNo, Tensor<double, 3u> const& gradE,
                  Tensor<double, 3u>& result) const;

    void jacobianInv(Tensor<double, 3u> const& jacobian, Tensor<double, 3u>& result) const;

    TensorBase<Vector<double>> detJResultInfo(std::size_t numPoints) const;
    void detJ(std::size_t eleNo, Tensor<double, 3u> const& jacobian,
              Tensor<double, 1u>& result) const;
    void absDetJ(std::size_t eleNo, Tensor<double, 3u> const& jacobian,
                 Tensor<double, 1u>& result) const;

    TensorBase<Matrix<double>> normalResultInfo(std::size_t numPoints) const;
    void normal(std::size_t faceNo, Tensor<double, 1u> const& detJ, Tensor<double, 3u> const& jInv,
                Tensor<double, 2u>& result) const;
    void normalize(Tensor<double, 2u>& normal) const;

    TensorBase<Tensor<double, 3u>> facetBasisResultInfo(std::size_t numPoints) const;
    void facetBasis(std::array<double, D> const& up, Matrix<double> const& normal,
                    Tensor<double, 3u>& result) const;
    void facetBasisFromPlexTangents(std::size_t faceNo, Tensor<double, 3u> const& jacobian,
                                    Matrix<double> const& normal, Tensor<double, 3u>& result) const;
    TensorBase<Vector<long int>> tagsInfo(std::size_t numPoints) const;
    void setVolumeTags(std::size_t eleNo, Tensor<long int, 1u>& result) const;
    void setFacetTags(FacetInfo const& info, Tensor<long int, 1u>& result) const;
    std::array<double, D> facetParam(std::size_t faceNo,
                                     std::array<double, D - 1> const& chi) const;

    std::vector<std::array<double, D>>
    facetParam(std::size_t faceNo, std::vector<std::array<double, D - 1>> const& chis) const;

    std::size_t numElements() const { return vertices.size(); }
    NodalRefElement<D> const& refElement() const { return refElement_; }

    /**
     * @brief Computes maximum mesh size on local partition
     *
     * The mesh size is defined as the maximum distance of the simplex's vertices.
     * That is, the mesh size is equal to the maximum diameter of the affine mesh approximating
     * the domain, thus, the maximum diameter of the "high-order" mesh might be larger.
     */
    double local_mesh_size() const { return local_mesh_size_; }

private:
    const unsigned N;
    NodalRefElement<D> refElement_;

    struct Verts {
        using type = std::array<double, D>;
    };

    mneme::StridedView<mneme::SingleStorage<Verts>> vertices;

    std::array<Simplex<D - 1>, D + 1> f2v;
    std::array<std::array<double, D>, D + 1> refVertices = Simplex<D>::referenceSimplexVertices();
    std::array<Eigen::Matrix<double, D, 1>, D + 1> refNormals;

    double local_mesh_size_;
    std::vector<long int> volumeTags;
    std::vector<long int> facetTags;
};

} // namespace tndm

#endif // CURVILINEAR_20200609_H
