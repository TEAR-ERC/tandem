#ifndef EIGENMAP_20200609_H
#define EIGENMAP_20200609_H

#include "Tensor.h"
#include "TensorBase.h"
#include "util/Utility.h"

#include <Eigen/Core>
#include <type_traits>

namespace tndm {

namespace detail {

template <typename Tensor, std::size_t D> struct EigenMapFactory;

template <typename Vector> struct EigenMapFactory<Vector, 1u> {

    auto operator()(Vector& vector) {
        using real_t = std::remove_pointer_t<decltype(vector.data())>;
        using type = Eigen::Map<copy_const<real_t, Eigen::VectorXd>>;
        return type(vector.data(), vector.shape(0));
    }
};

template <typename Matrix> struct EigenMapFactory<Matrix, 2u> {
    auto operator()(Matrix& matrix) {
        using real_t = std::remove_pointer_t<decltype(matrix.data())>;
        using type = Eigen::Map<copy_const<real_t, Eigen::MatrixXd>>;
        return type(matrix.data(), matrix.shape(0), matrix.shape(1));
    }
};

} // namespace detail

template <typename Tensor> auto EigenMap(Tensor& tensor) {
    detail::EigenMapFactory<Tensor, detail::traits<Tensor>::Dim> factory;
    return factory(tensor);
}

} // namespace tndm

#endif // EIGENMAP_20200609_H
