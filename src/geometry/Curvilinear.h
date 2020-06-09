#ifndef CURVILINEAR_20200609_H
#define CURVILINEAR_20200609_H

#include "basis/Nodal.h"
#include "basis/WarpAndBlend.h"
#include "mesh/LocalSimplexMesh.h"
#include "tensor/Matrix.h"
#include "tensor/TensorView.h"
#include "tensor/Vector.h"

#include <mneme/storage.hpp>
#include <mneme/view.hpp>

#include <Eigen/Dense>

#include <array>
#include <functional>

namespace tndm {

template <std::size_t D> class Curvilinear {
public:
    using vertex_t = std::array<double, D>;

    Curvilinear(
        LocalSimplexMesh<D> const& mesh,
        std::function<vertex_t(vertex_t const&)> transform = [](vertex_t const& v) { return v; },
        unsigned degree = 1, NodesFactory<D> const& nodesFactory = WarpAndBlendFactory<D>());

    Matrix<double> evaluateBasisAt(std::vector<std::array<double, D>> const& points);
    Tensor<double, 3u> evaluateGradientAt(std::vector<std::array<double, D>> const& points);

    TensorInfo<Matrix<double>> mapResultInfo(std::size_t numPoints) const;
    void map(std::size_t eleNo, Matrix<double> const& E, TensorView<Tensor<double, 2u>>& result);

    TensorInfo<Tensor<double, 3u>> jacobianResultInfo(std::size_t numPoints) const;
    void jacobian(std::size_t eleNo, Tensor<double, 3u> const& gradE,
                  TensorView<Tensor<double, 3u>>& result);

    void jacobianInvT(TensorView<Tensor<double, 3u>> const& jacobian,
                      TensorView<Tensor<double, 3u>>& result);

    TensorInfo<Vector<double>> detJResultInfo(std::size_t numPoints) const;
    void detJ(std::size_t eleNo, TensorView<Tensor<double, 3u>> const& jacobian,
              TensorView<Tensor<double, 1u>>& result);

    TensorInfo<Matrix<double>> normalResultInfo(std::size_t numPoints) const;
    void normal(std::size_t faceNo, TensorView<Tensor<double, 1u>> const& detJ,
                TensorView<Tensor<double, 3u>> const& JinvT,
                TensorView<Tensor<double, 2u>>& result);

    std::array<double, D> facetParam(std::size_t faceNo, std::array<double, D - 1> const& chi);

private:
    const unsigned N;

    struct Verts {
        using type = std::array<double, D>;
    };

    mneme::SingleStorage<Verts> storage;
    mneme::StridedView<mneme::SingleStorage<Verts>> vertices;

    std::array<Simplex<D - 1>, D + 1> f2v;
    std::array<std::array<double, D>, D + 1> refVertices = Simplex<D>::referenceSimplexVertices();
    std::array<Eigen::Matrix<double, D, 1>, D + 1> refNormals;

    Eigen::MatrixXd vandermonde;
    Eigen::MatrixXd vandermondeInvT;
};

} // namespace tndm

#endif // CURVILINEAR_20200609_H
