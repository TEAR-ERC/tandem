#ifndef ABSTRACTADAPTEROPERATOR_20210910_H
#define ABSTRACTADAPTEROPERATOR_20210910_H

#include "form/BoundaryMap.h"
#include "interface/BlockVector.h"
#include "interface/BlockView.h"

#include <mpi.h>

#include <functional>

namespace tndm {

class AbstractAdapterOperator {
public:
    virtual ~AbstractAdapterOperator() {}

    virtual BoundaryMap const& fault_map() const = 0;
    virtual std::size_t num_local_elements() const = 0;
    virtual std::size_t traction_block_size() const = 0;
    virtual MPI_Comm comm() const = 0;

    virtual auto slip_bc(BlockView const& state)
        -> std::function<void(std::size_t, Matrix<double>&, bool)> = 0;
    virtual void traction(BlockView const& displacement, BlockVector& result) = 0;
};

} // namespace tndm

#endif // ABSTRACTADAPTEROPERATOR_20210910_H
