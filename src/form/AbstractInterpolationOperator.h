#ifndef ABSTRACTINTERPOLATIONOPERATOR_20210910_H
#define ABSTRACTINTERPOLATIONOPERATOR_20210910_H

#include "interface/BlockMatrix.h"

namespace tndm {

class AbstractInterpolationOperator {
public:
    virtual ~AbstractInterpolationOperator() {}

    virtual unsigned max_degree() const = 0;
    virtual std::size_t block_size(unsigned degree) const = 0;
    virtual void assemble(unsigned to_degree, unsigned from_degree, BlockMatrix& matrix) = 0;
};

} // namespace tndm

#endif // ABSTRACTINTERPOLATIONOPERATOR_20210910_H
