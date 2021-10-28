#ifndef ABSTRACTDGOPERATOR_20210910_H
#define ABSTRACTDGOPERATOR_20210910_H

#include "form/AbstractInterpolationOperator.h"
#include "form/DGOperatorTopo.h"
#include "form/FiniteElementFunction.h"
#include "interface/BlockMatrix.h"
#include "interface/BlockVector.h"
#include "tensor/Tensor.h"
#include "util/Range.h"

#include <cstddef>
#include <functional>
#include <optional>

namespace tndm {

template <std::size_t D> class AbstractDGOperator {
public:
    using volume_functional_t = std::function<void(std::size_t elNo, Matrix<double>& F)>;
    using facet_functional_t =
        std::function<void(std::size_t fctNo, Matrix<double>& f, bool is_boundary)>;

    virtual ~AbstractDGOperator() {}

    virtual std::size_t block_size() const = 0;
    virtual std::size_t num_local_elements() const = 0;
    virtual std::size_t number_of_local_dofs() const { return block_size() * num_local_elements(); }
    virtual DGOperatorTopo const& topo() const = 0;

    virtual auto interpolation_operator() -> std::unique_ptr<AbstractInterpolationOperator> = 0;
    virtual void assemble(BlockMatrix& matrix) = 0;
    virtual void rhs(BlockVector& vector) = 0;
    virtual void apply(BlockVector const& x, BlockVector& y) = 0;
    virtual std::size_t flops_apply() const = 0;
    virtual void wave_rhs(BlockVector const& x, BlockVector& y) = 0;
    virtual void project(volume_functional_t x, BlockVector& y) = 0;
    virtual double local_cfl_time_step() const = 0;

    virtual auto solution(BlockVector const& vector, std::vector<std::size_t> const& subset)
        -> FiniteElementFunction<D> = 0;
    virtual auto solution(BlockVector const& vector,
                          std::optional<Range<std::size_t>> range = std::nullopt)
        -> FiniteElementFunction<D> = 0;
    virtual auto params(std::vector<std::size_t> const& subset) -> FiniteElementFunction<D> = 0;
    virtual auto params(std::optional<Range<std::size_t>> range = std::nullopt)
        -> FiniteElementFunction<D> = 0;

    virtual void set_force(volume_functional_t fun) = 0;
    virtual void set_slip(facet_functional_t fun) = 0;
    virtual void set_dirichlet(facet_functional_t fun) = 0;
};

} // namespace tndm

#endif // ABSTRACTDGOPERATOR_20210910_H
