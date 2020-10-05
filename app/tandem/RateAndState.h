#ifndef RATEANDSTATE_20201001_H
#define RATEANDSTATE_20201001_H

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

class RateAndState {
public:
    constexpr static std::size_t NumQuantities = 2;

    RateAndState(Curvilinear<DomainDimension> const& cl);

    std::size_t block_size() const { return space_.numBasisFunctions() * NumQuantities; }
    std::size_t scratch_mem_size() const { return 1; }

    void begin_preparation(std::size_t numFaultFaces);
    void prepare(std::size_t faultNo, FacetInfo const& info, LinearAllocator& scratch);
    void end_preparation() {}

    void initial(std::size_t faultNo, Vector<double>& B, LinearAllocator& scratch) const;
    void rhs(std::size_t faultNo, Vector<double>& B, LinearAllocator& scratch) const;

    auto state_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension - 1u>(space_.clone(), NumQuantities,
                                                           numLocalElements);
    }

private:
    Curvilinear<DomainDimension> const* cl_;
    NodalRefElement<DomainDimension - 1u> space_;

    // Basis
    std::vector<Managed<Matrix<double>>> geoE_q;
    std::vector<Managed<Tensor<double, 3u>>> geoDxi_q;

    struct Coords {
        using type = std::array<double, DomainDimension>;
    };

    using fault_t = mneme::MultiStorage<mneme::DataLayout::SoA, Coords>;
    mneme::StridedView<fault_t> fault_;
};

} // namespace tndm

#endif // RATEANDSTATE_20201001_H
