#ifndef DG_20200615_H
#define DG_20200615_H

#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "mesh/LocalSimplexMesh.h"
#include "quadrules/AutoRule.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include <mneme/storage.hpp>
#include <mneme/view.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

template <std::size_t D> class DG {
public:
    DG(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl,
       std::unique_ptr<RefElement<D>> refElement, unsigned minQuadOrder);
    virtual ~DG() = default;

    std::size_t numElements() const { return vol.size(); }
    std::size_t numFacets() const { return fctInfo.size(); }

    RefElement<D> const& refElement() const { return *refElement_; }

protected:
    void facetPrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl);
    void volumePrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl);

    std::unique_ptr<RefElement<D>> refElement_;

    SimplexQuadratureRule<D - 1u> fctRule;
    SimplexQuadratureRule<D> volRule;

    Managed<Matrix<double>> E;
    Managed<Tensor<double, 3u>> D_xi;
    std::vector<Managed<Matrix<double>>> e;
    std::vector<Managed<Tensor<double, 3u>>> d_xi;

    struct AbsDetJ {
        using type = double;
    };
    struct JInv {
        using type = std::array<double, D * D>;
    };
    struct JInvOther {
        using type = std::array<double, D * D>;
    };
    struct Normal {
        using type = std::array<double, D>;
    };
    struct NormalLength {
        using type = double;
    };
    struct Coords {
        using type = std::array<double, D>;
    };

    struct FacetInfo {
        std::array<std::size_t, 2> up;
        std::array<std::size_t, 2> localNo;
        double area;
    };
    std::vector<FacetInfo> fctInfo;
    std::vector<double> penalty_;

    using fct_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, JInv, JInvOther, Normal, NormalLength, Coords>;
    mneme::StridedView<fct_t> fct;

    using vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, AbsDetJ, JInv, Coords>;
    mneme::StridedView<vol_t> vol;
};

} // namespace tndm

#endif // DG_20200615_H
