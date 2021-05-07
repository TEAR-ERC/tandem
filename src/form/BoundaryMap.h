#ifndef BOUNDARYMAP_20201005_H
#define BOUNDARYMAP_20201005_H

#include "form/BC.h"
#include "form/DGOperatorTopo.h"
#include "mesh/LocalSimplexMesh.h"
#include "parallel/ScatterPlan.h"

#include <cstddef>
#include <memory>
#include <mpi.h>
#include <vector>

namespace tndm {

class BoundaryMap {
public:
    template <std::size_t D>
    BoundaryMap(LocalSimplexMesh<D> const& mesh, BC bc, MPI_Comm comm = MPI_COMM_WORLD);

    std::size_t fctNo(std::size_t bndNo) const { return fctNos_[bndNo]; }
    std::vector<std::size_t> localFctNos() const {
        return std::vector<std::size_t>(fctNos_.begin(), fctNos_.begin() + local_size_);
    }
    std::vector<std::size_t> const& fctNos() const { return fctNos_; }
    std::size_t bndNo(std::size_t fctNo) const { return bndNos_[fctNo]; }

    std::size_t size() const { return fctNos_.size(); }
    std::size_t local_size() const { return local_size_; }
    std::shared_ptr<ScatterPlan> scatter_plan() const { return scatter_plan_; }

private:
    std::vector<std::size_t> fctNos_;
    std::vector<std::size_t> bndNos_;
    std::size_t local_size_;
    std::shared_ptr<ScatterPlan> scatter_plan_;
};

} // namespace tndm

#endif // BOUNDARYMAP_20201005_H
