#ifndef DGCURVILINEARCOMMON_20200911_H
#define DGCURVILINEARCOMMON_20200911_H

#include "form/FacetInfo.h"
#include "geometry/Curvilinear.h"
#include "geometry/Vector.h"
#include "parallel/Scatter.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <vector>

namespace tndm {

template <std::size_t D> class DGCurvilinearCommon {
public:
    template <std::size_t Q>
    using functional_t = std::function<std::array<double, Q>(std::array<double, D> const&)>;
    using volume_functional_t = std::function<void(std::size_t elNo, Matrix<double>& F)>;
    using facet_functional_t =
        std::function<void(std::size_t fctNo, Matrix<double>& f, bool is_boundary)>;

    DGCurvilinearCommon(Curvilinear<D> const& cl, unsigned minQuadOrder);

    std::size_t scratch_mem_size() const {
        return std::max(volRule.size(), fctRule.size()) * sizeof(double) * (D * D + 1);
    }

    void begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                           std::size_t numLocalFacets);
    void prepare_skeleton(std::size_t fctNo, FacetInfo const& info, LinearAllocator& scratch) {
        prepare_bndskl(fctNo, info, false, scratch);
    }
    void prepare_boundary(std::size_t fctNo, FacetInfo const& info, LinearAllocator& scratch) {
        prepare_bndskl(fctNo, info, true, scratch);
    }
    void prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator& scratch);
    void end_preparation(Scatter& elementScatter);

    template <std::size_t Q>
    auto make_volume_functional(functional_t<Q> fun) const -> volume_functional_t {
        return [fun, this](std::size_t elNo, Matrix<double>& F) {
            assert(Q == F.shape(0));
            auto coords = this->vol[elNo].template get<Coords>();
            for (std::size_t q = 0; q < F.shape(1); ++q) {
                auto fx = fun(coords[q]);
                for (std::size_t p = 0; p < F.shape(0); ++p) {
                    F(p, q) = fx[p];
                }
            }
        };
    }

    template <std::size_t Q>
    auto make_facet_functional(functional_t<Q> fun) const -> facet_functional_t {
        return [fun, this](std::size_t fctNo, Matrix<double>& f, bool) {
            assert(Q == f.shape(0));
            auto coords = this->fct[fctNo].template get<Coords>();
            for (std::size_t q = 0; q < f.shape(1); ++q) {
                auto fx = fun(coords[q]);
                for (std::size_t p = 0; p < f.shape(0); ++p) {
                    f(p, q) = fx[p];
                }
            }
        };
    }
    template <std::size_t Q>
    auto make_facet_functional(functional_t<Q> fun, std::array<double, D> const& refNormal) const
        -> facet_functional_t {
        return [fun, refNormal, this](std::size_t fctNo, Matrix<double>& f, bool is_boundary) {
            assert(Q == f.shape(0));
            auto coords = this->fct[fctNo].template get<Coords>();
            for (std::size_t q = 0; q < f.shape(1); ++q) {
                auto fx = fun(coords[q]);
                if (!is_boundary) {
                    auto normal = this->fct[fctNo].template get<Normal>()[q];
                    if (dot(refNormal, normal) < 0) {
                        fx = -1.0 * fx;
                    }
                }
                for (std::size_t p = 0; p < f.shape(0); ++p) {
                    f(p, q) = fx[p];
                }
            }
        };
    }

    static void zero_volume_function(std::size_t, Matrix<double>& x) { x.set_zero(); }
    static void zero_facet_function(std::size_t, Matrix<double>& x, bool) { x.set_zero(); }

    SimplexQuadratureRule<D - 1u> const& facetQuadratureRule() const { return fctRule; }
    SimplexQuadratureRule<D> const& volQuadratureRule() const { return volRule; }

protected:
    void prepare_bndskl(std::size_t fctNo, FacetInfo const& info, bool isBnd,
                        LinearAllocator& scratch);

    Curvilinear<D> const* cl_;

    // Rules
    SimplexQuadratureRule<D - 1u> fctRule;
    SimplexQuadratureRule<D> volRule;

    // Basis
    Managed<Matrix<double>> geoE_Q;
    Managed<Tensor<double, 3u>> geoDxi_Q;
    std::vector<Managed<Matrix<double>>> geoE_q;
    std::vector<Managed<Tensor<double, 3u>>> geoDxi_q;

    // Precomputed data
    struct AbsDetJ {
        using type = double;
    };
    struct JInv {
        using type = std::array<double, D * D>;
    };
    struct JInv0 {
        using type = std::array<double, D * D>;
    };
    struct JInv1 {
        using type = std::array<double, D * D>;
    };
    struct Normal {
        using type = std::array<double, D>;
    };
    struct UnitNormal {
        using type = std::array<double, D>;
    };
    struct NormalLength {
        using type = double;
    };
    struct Coords {
        using type = std::array<double, D>;
    };

    using fct_t = mneme::MultiStorage<mneme::DataLayout::SoA, JInv0, JInv1, Normal, UnitNormal,
                                      NormalLength, Coords>;
    using vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, AbsDetJ, JInv, Coords>;

    mneme::StridedView<fct_t> fct;
    mneme::StridedView<vol_t> vol;
    std::vector<double> penalty;
};

} // namespace tndm

#endif // DGCURVILINEARCOMMON_20200911_H
