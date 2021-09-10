#ifndef BLOCKMATRIX_20210910_H
#define BLOCKMATRIX_20210910_H

#include "tensor/Tensor.h"

#include <cstddef>

namespace tndm {

class BlockMatrix {
public:
    using value_type = double;

    virtual ~BlockMatrix() {}

    virtual void add_block(std::size_t ib_local, std::size_t jb_local,
                           Matrix<double> const& values) = 0;
    virtual void begin_assembly() = 0;
    virtual void end_assembly() = 0;

    virtual void set_zero() = 0;
};

} // namespace tndm

#endif // BLOCKMATRIX_20210910_H
