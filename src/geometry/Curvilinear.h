#ifndef CURVILINEAR_H
#define CURVILINEAR_H

#include "mesh/LocalSimplexMesh.h"

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
        bool quadratic = false);

    std::array<double, D> map(std::size_t eleNo, std::array<double, D> const& xi);
    std::array<double, D * D> jacobian(std::size_t eleNo, std::array<double, D> const& xi);
    double detJ(std::size_t eleNo, std::array<double, D> const& xi);
    std::array<double, D> normal(std::size_t eleNo, std::size_t faceNo,
                                 std::array<double, D> const& xi);

    std::array<double, D> facetParam(std::size_t faceNo, std::array<double, D - 1> const& chi);

private:
    std::size_t degree;

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

#endif // CURVILINEAR_H
