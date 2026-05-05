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
    static auto Space() -> NodalRefElement<DomainDimension - 1u>;

    static constexpr std::size_t TangentialComponents = DomainDimension - 1u;
    // D-1 slip components + 1 state component
    constexpr static std::size_t NumQuantities = TangentialComponents + 1u;

    RateAndStateBase(std::shared_ptr<Curvilinear<DomainDimension>> cl);

    std::size_t block_size() const { return space_.numBasisFunctions() * NumQuantities; }
    std::size_t slip_block_size() const {
        return space_.numBasisFunctions() * TangentialComponents;
    }
    std::size_t scratch_mem_size() const { return 1; }

    void begin_preparation(std::size_t numFaultFaces);
    void prepare(std::size_t faultNo, FacetInfo const& info, LinearAllocator<double>&);
    void end_preparation() {}

    auto const& space() const { return space_; }
    auto const& cl() const { return cl_; }

    /// Returns physical coordinates of node @p node on fault element @p faultNo.
    std::array<double, DomainDimension> fault_node_coords(std::size_t faultNo,
                                                          std::size_t node) const {
        return fault_[faultNo].template get<Coords>()[node];
    }

    /// Fills @p out with all fault node physical coordinates, interleaved as
    /// [faultNo0_node0_x, faultNo0_node0_y, ..., faultNo0_nodeN_x, faultNo0_nodeN_y, ...,
    ///  faultNo1_node0_x, ...].
    void fill_fault_node_coords(std::vector<double>& out) const {
        auto const nbf = space_.numBasisFunctions();
        auto const nfault = fault_.size();
        out.resize(nfault * nbf * DomainDimension);
        std::size_t offset = 0;
        for (std::size_t faultNo = 0; faultNo < nfault; ++faultNo) {
            auto coords = fault_[faultNo].template get<Coords>();
            for (std::size_t n = 0; n < nbf; ++n) {
                auto const& x = coords[n];
                for (std::size_t d = 0; d < DomainDimension; ++d) {
                    out[offset++] = x[d];
                }
            }
        }
    }

protected:
    std::shared_ptr<Curvilinear<DomainDimension>> cl_;

    // Basis
    NodalRefElement<DomainDimension - 1u> space_;
    std::vector<Managed<Matrix<double>>> geoE_q;

    struct Coords {
        using type = std::array<double, DomainDimension>;
    };
    using fault_t = mneme::MultiStorage<mneme::DataLayout::SoA, Coords>;
    mneme::StridedView<fault_t> fault_;
};

} // namespace tndm

#endif // RATEANDSTATEBASE_20201026_H
