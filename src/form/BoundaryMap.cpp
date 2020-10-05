#include "BoundaryMap.h"
#include "form/DGOperatorTopo.h"

#include <limits>

namespace tndm {
BoundaryMap::BoundaryMap(DGOperatorTopo const& topo, BC bc) {
    std::size_t numFacets = topo.numLocalFacets();
    fctNos_.reserve(numFacets);
    bndNos_.resize(numFacets, std::numeric_limits<std::size_t>::max());
    for (std::size_t fctNo = 0; fctNo < numFacets; ++fctNo) {
        auto const& info = topo.info(fctNo);
        if (info.bc == bc) {
            bndNos_[fctNo] = fctNos_.size();
            fctNos_.push_back(fctNo);
        }
    }
    fctNos_.shrink_to_fit();
}

} // namespace tndm
