#ifndef DG_20200615_H
#define DG_20200615_H

#include "mesh/MeshData.h"
#include "parallel/Scatter.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include <mneme/storage.hpp>
#include <mneme/view.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

template <std::size_t D> class Curvilinear;
template <std::size_t D> class LocalSimplexMesh;
template <std::size_t D> class RefElement;


template <std::size_t D> class DG {
public:
    DG(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl,
       std::unique_ptr<RefElement<D>> refElement, unsigned minQuadOrder, MPI_Comm comm);
    virtual ~DG() = default;

    std::size_t numElements() const { return numElems_; }
    std::size_t numLocalElements() const { return numLocalElems_; }
    std::size_t numLocalFacets() const { return numLocalFacets_; }

    RefElement<D> const& refElement() const { return *refElement_; }
    auto const& facetRule() const { return fctRule; }
    auto const& volumeRule() const { return volRule; }

    MPI_Comm comm() const { return comm_; }

protected:
    void facetPrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl);
    void volumePrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl);

    std::unique_ptr<RefElement<D>> refElement_;
    std::size_t numElems_;
    std::size_t numLocalElems_;
    std::size_t numLocalFacets_;
    MPI_Comm comm_;
    Scatter elementScatter_;

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
        std::array<bool, 2> inside;
        std::array<std::size_t, 2> up;
        std::array<std::size_t, 2> g_up;
        std::array<std::size_t, 2> localNo;
        double area;
        BC bc;
    };
    std::vector<FacetInfo> fctInfo;

    struct GID {
        using type = std::size_t;
    };
    struct Penalty {
        using type = double;
    };
    struct NumLocalNeighbours {
        using type = unsigned;
    };
    struct NumGhostNeighbours {
        using type = unsigned;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, GID, Penalty, NumLocalNeighbours,
                        NumGhostNeighbours>
        volInfo;

    using fct_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, JInv, JInvOther, Normal, NormalLength, Coords>;
    mneme::StridedView<fct_t> fct;

    using vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, AbsDetJ, JInv, Coords>;
    mneme::StridedView<vol_t> vol;
};

} // namespace tndm

#endif // DG_20200615_H
