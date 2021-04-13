#ifndef BLOCKVECTOR_20210413_H
#define BLOCKVECTOR_20210413_H

#include "tensor/Tensor.h"

#include <cstddef>

namespace tndm {

class BlockVector {
public:
    using value_type = double;

    virtual ~BlockVector() {}

    virtual std::size_t block_size() const = 0;
    virtual void begin_assembly() = 0;
    virtual void add_block(std::size_t ib_local, Vector<double> const& values) = 0;
    virtual void add_block(std::size_t ib_local, Vector<const double> const& values) = 0;
    virtual void insert_block(std::size_t ib_local, Vector<double> const& values) = 0;
    virtual void insert_block(std::size_t ib_local, Vector<const double> const& values) = 0;
    virtual void end_assembly() = 0;
    virtual void set_zero() = 0;

    virtual Matrix<double> begin_access() = 0;
    virtual void end_access(Matrix<double>& data) = 0;
    virtual Matrix<const double> begin_access_readonly() const = 0;
    virtual void end_access_readonly(Matrix<const double>& data) const = 0;
};

} // namespace tndm

#endif // BLOCKVECTOR_20210413_H
