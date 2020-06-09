#ifndef MATRIX_20200609_H
#define MATRIX_20200609_H

#include "Tensor.h"

namespace tndm {

template <typename real_t> class Matrix : public Tensor<real_t, 2u> {
public:
    using Base = Tensor<real_t, 2u>;
    using typename Base::index_t;

    Matrix(index_t rows, index_t cols) : Base({rows, cols}) {}

    index_t rows() const { return Base::shape(0); }
    index_t cols() const { return Base::shape(1); }
};

namespace detail {
template <typename real_t> struct traits<Matrix<real_t>> : traits<Tensor<real_t, 2u>> {};
} // namespace detail

} // namespace tndm

#endif // MATRIX_20200609_H
