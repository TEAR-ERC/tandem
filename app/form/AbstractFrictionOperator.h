#ifndef ABSTRACTFRICTIONOPERATOR_20210910_H
#define ABSTRACTFRICTIONOPERATOR_20210910_H

#include "config.h"

#include "form/FiniteElementFunction.h"
#include "interface/BlockVector.h"
#include "util/Range.h"

#include <cstddef>
#include <optional>

namespace tndm {

class AbstractFrictionOperator {
public:
    virtual ~AbstractFrictionOperator() {}

    virtual std::size_t block_size() const = 0;
    virtual std::size_t slip_block_size() const = 0;
    virtual std::size_t num_local_elements() const = 0;
    virtual std::size_t number_of_local_dofs() const { return num_local_elements() * block_size(); }
    virtual double VMax_local() const = 0;

    virtual void pre_init(BlockVector& state) = 0;
    virtual void init(double time, BlockVector const& traction, BlockVector& state) = 0;
    virtual void rhs(double time, BlockVector const& traction, BlockVector const& state,
                     BlockVector& result) = 0;

    virtual auto state(double time, BlockVector const& traction, BlockVector const& state,
                       std::vector<std::size_t> const& subset)
        -> FiniteElementFunction<DomainDimension - 1u> = 0;
    virtual auto state(double time, BlockVector const& traction, BlockVector const& state,
                       std::optional<Range<std::size_t>> range = std::nullopt)
        -> FiniteElementFunction<DomainDimension - 1u> = 0;
    virtual auto raw_state(BlockVector const& state)
        -> FiniteElementFunction<DomainDimension - 1u> = 0;
};

} // namespace tndm

#endif // ABSTRACTFRICTIONOPERATOR_20210910_H
