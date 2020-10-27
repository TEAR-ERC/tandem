#ifndef RATEANDSTATEBASE_20201026_H
#define RATEANDSTATEBASE_20201026_H

#include "config.h"

#include "form/FacetInfo.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class RateAndStateBase {
public:
    constexpr static std::size_t NumQuantities = 2;
    constexpr static std::size_t NumInternalQuantities = 4;

    RateAndStateBase(Curvilinear<DomainDimension> const& cl,
                     std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints);

    std::size_t block_size() const { return space_.numBasisFunctions() * NumQuantities; }
    std::size_t scratch_mem_size() const {
        return space_.numBasisFunctions() * (2 * DomainDimension * DomainDimension + 1) *
               sizeof(double);
    }

    void begin_preparation(std::size_t numFaultFaces);
    void prepare(std::size_t faultNo, FacetInfo const& info,
                 std::array<double, DomainDimension> const& ref_normal, LinearAllocator& scratch);
    void end_preparation() {}

    void slip(std::size_t faultNo, Vector<double const>& state, Matrix<double>& result,
              LinearAllocator& scratch) const;

    auto state_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension - 1u>(space_.clone(), NumInternalQuantities,
                                                           numLocalElements);
    }

protected:
    void compute_traction(std::size_t faultNo, Matrix<double> const& grad_u,
                          Vector<double>& traction) const;

    Curvilinear<DomainDimension> const* cl_;
    NodalRefElement<DomainDimension - 1u> space_;

    // Basis
    Managed<Matrix<double>> e_q_T;
    std::vector<Managed<Matrix<double>>> geoE_q;
    std::vector<Managed<Tensor<double, 3u>>> geoDxi_q;

    struct Coords {
        using type = std::array<double, DomainDimension>;
    };
    struct UnitNormal {
        using type = std::array<double, DomainDimension>;
    };
    struct SignFlipped {
        using type = bool;
    };

    using fault_t = mneme::MultiStorage<mneme::DataLayout::SoA, Coords, UnitNormal, SignFlipped>;
    mneme::StridedView<fault_t> fault_;
};

} // namespace tndm

#endif // RATEANDSTATEBASE_20201026_H
