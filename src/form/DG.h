#ifndef DG_20200615_H
#define DG_20200615_H

#include "geometry/Curvilinear.h"
#include "mesh/LocalSimplexMesh.h"
#include "quadrules/AutoRule.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include <mneme/storage.hpp>
#include <mneme/view.hpp>

#include <cstddef>
#include <vector>

namespace tndm {

template <std::size_t D> class DG {
public:
    DG(LocalSimplexMesh<D> const& mesh, Curvilinear<D> const& cl, unsigned minQuadOrder);
    virtual ~DG() = default;

private:
    void facetPrecompute(LocalSimplexMesh<D> const& mesh);
    void volumePrecompute(LocalSimplexMesh<D> const& mesh);

    Curvilinear<D> const& cl;
    SimplexQuadratureRule<D - 1u> fctRule;
    SimplexQuadratureRule<D> volRule;

    Managed<Matrix<double>> E;
    Managed<Matrix<double>> gradE;
    std::vector<Managed<Matrix<double>>> fctE;
    std::vector<Managed<Matrix<double>>> fctGradE;

    struct detJ {
        using type = double;
    };
    struct JinvT {
        using type = std::array<double, D*D>;
    };
    struct normal {
        using type = std::array<double, D>;
    };

    struct FacetInfo {
        std::array<std::size_t, 2> up;
        std::array<std::size_t, 2> localNo;
    };
    std::vector<FacetInfo> fctInfo;

    mneme::MultiStorage<mneme::DataLayout::SoA, detJ, JinvT, normal> fctStore;
    mneme::StridedView<decltype(fctStore)> fct;

    mneme::MultiStorage<mneme::DataLayout::SoA, detJ, JinvT> volStore;
    mneme::StridedView<decltype(volStore)> vol;
};

} // namespace tndm

#endif // DG_20200615_H
