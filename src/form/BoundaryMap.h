#ifndef BOUNDARYMAP_20201005_H
#define BOUNDARYMAP_20201005_H

#include "form/BC.h"
#include "form/DGOperatorTopo.h"

#include <cstddef>
#include <vector>

namespace tndm {

class BoundaryMap {
public:
    BoundaryMap(DGOperatorTopo const& topo, BC bc);

    std::size_t fctNo(std::size_t bndNo) const { return fctNos_[bndNo]; }
    std::size_t bndNo(std::size_t fctNo) const { return bndNos_[fctNo]; }

    std::size_t size() const { return fctNos_.size(); }

private:
    std::vector<std::size_t> fctNos_;
    std::vector<std::size_t> bndNos_;
};

} // namespace tndm

#endif // BOUNDARYMAP_20201005_H
