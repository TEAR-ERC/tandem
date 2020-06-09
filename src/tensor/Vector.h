#ifndef VECTOR_20200609_H
#define VECTOR_20200609_H

#include "Tensor.h"

namespace tndm {

template <typename real_t> class Vector : public Tensor<real_t, 1u> {
public:
    using Base = Tensor<real_t, 1u>;
    using typename Base::index_t;

    Vector(index_t size) : Base({size}) {}

    index_t size() const { return Base::shape(0); }
};

namespace detail {
template <typename real_t> struct traits<Vector<real_t>> : traits<Tensor<real_t, 1u>> {};
} // namespace detail

} // namespace tndm

#endif // VECTOR_20200609_H
