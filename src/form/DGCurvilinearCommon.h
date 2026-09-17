#ifndef DGCURVILINEARCOMMON_20200911_H
#define DGCURVILINEARCOMMON_20200911_H

#include "form/FacetInfo.h"
#include "geometry/Curvilinear.h"
#include "geometry/Vector.h"
#include "parallel/ScatterPlan.h"
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
#include <memory>
#include <vector>

namespace tndm {

enum class DGMethod { IP, BR2, Unknown };

template <std::size_t D> class DGCurvilinearCommon {
public:
    template <std::size_t Q>
    using functional_t =
        std::function<std::array<double, Q>(std::array<double, D> const&, long int)>;
    using volume_functional_t = std::function<void(std::size_t elNo, Matrix<double>& F)>;
    using facet_functional_t =
        std::function<void(std::size_t fctNo, Matrix<double>& f, bool is_boundary)>;

    constexpr static std::size_t NumFacets = D + 1;

    DGCurvilinearCommon(std::shared_ptr<Curvilinear<D>> cl, unsigned minQuadOrder);

    Curvilinear<D> const& cl() const { return *cl_; }
    std::shared_ptr<Curvilinear<D>> cl_ptr() const { return cl_; }

    std::size_t scratch_mem_size() const {
        return std::max(volRule.size(), fctRule.size()) * (D * D + 1);
    }

    void begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                           std::size_t numLocalFacets);
    void prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch) {
        prepare_bndskl(fctNo, info, scratch);
    }
    void prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch) {
        prepare_bndskl(fctNo, info, scratch);
    }
    void prepare_volume_post_skeleton(std::size_t, LinearAllocator<double>&) {}
    void end_preparation(std::shared_ptr<ScatterPlan> elementScatterPlan);

    template <std::size_t Q>
    auto make_volume_functional(functional_t<Q> fun) const -> volume_functional_t {
        return [fun, this](std::size_t elNo, Matrix<double>& F) {
            assert(Q == F.shape(0));
            auto coords = this->vol[elNo].template get<Coords>();
            long int tag = cl_->getVolumeTag(elNo);
            for (std::size_t q = 0; q < F.shape(1); ++q) {
                auto fx = fun(coords[q], tag);
                for (std::size_t p = 0; p < F.shape(0); ++p) {
                    F(p, q) = fx[p];
                }
            }
        };
    }

    /**
     * @brief make_volume_functional for evaluation points other than the quadrature points
     *
     * The overload above is tied to volRule.points(), because it reads the pre-computed
     * coordinates of the quadrature points. This one maps an arbitrary set of reference points
     * to physical space itself, which is what nodal interpolation of a coefficient needs.
     *
     * @param fun Function to evaluate
     * @param refPoints Evaluation points in the reference element
     */
    template <std::size_t Q>
    auto make_volume_functional(functional_t<Q> fun,
                                std::vector<std::array<double, D>> const& refPoints) const
        -> volume_functional_t {
        auto geoE = std::make_shared<Managed<Matrix<double>>>(cl_->evaluateBasisAt(refPoints));
        auto const numPoints = refPoints.size();
        return [fun, geoE, numPoints, this](std::size_t elNo, Matrix<double>& F) {
            assert(Q == F.shape(0));
            assert(numPoints == F.shape(1));
            // Allocated here rather than captured so that the functional stays re-entrant
            auto coords = Managed<Matrix<double>>(cl_->mapResultInfo(numPoints));
            cl_->map(elNo, *geoE, coords);
            long int tag = cl_->getVolumeTag(elNo);
            auto x = std::array<double, D>{};
            for (std::size_t q = 0; q < numPoints; ++q) {
                for (std::size_t d = 0; d < D; ++d) {
                    x[d] = coords(d, q);
                }
                auto fx = fun(x, tag);
                for (std::size_t p = 0; p < Q; ++p) {
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
            auto tag = cl_->getFacetTag(fctNo);

            for (std::size_t q = 0; q < f.shape(1); ++q) {
                auto fx = fun(coords[q], tag);
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
            auto tag = cl_->getFacetTag(fctNo);
            for (std::size_t q = 0; q < f.shape(1); ++q) {
                auto fx = fun(coords[q], tag);
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
    static void one_volume_function(std::size_t, Matrix<double>& x) { x.set_constant(1.0); }
    static void zero_facet_function(std::size_t, Matrix<double>& x, bool) { x.set_zero(); }

    SimplexQuadratureRule<D - 1u> const& facetQuadratureRule() const { return fctRule; }
    SimplexQuadratureRule<D> const& volQuadratureRule() const { return volRule; }

protected:
    void prepare_bndskl(std::size_t fctNo, FacetInfo const& info, LinearAllocator<double>& scratch);

    std::shared_ptr<Curvilinear<D>> cl_;

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
    std::vector<double> area_;
    std::vector<double> volume_;
};

} // namespace tndm

#endif // DGCURVILINEARCOMMON_20200911_H
