#ifndef EIGENMAP_20200609_H
#define EIGENMAP_20200609_H

#include "Tensor.h"
#include "TensorBase.h"
#include "util/Utility.h"

#include <Eigen/Core>
#include <type_traits>

namespace tndm {

namespace detail {

template <typename Tensor, int Rows, int Cols, std::size_t D> struct EigenMapFactory;

template <typename Vector, int Rows, int Cols> struct EigenMapFactory<Vector, Rows, Cols, 1u> {
    auto operator()(Vector& vector) {
        using real_t = std::remove_pointer_t<decltype(vector.data())>;
        using vector_t = copy_const<real_t, Eigen::Matrix<std::decay_t<real_t>, Rows, 1>>;
        if constexpr (detail::traits<Vector>::Packed) {
            return Eigen::Map<vector_t>(vector.data(), vector.shape(0));
        } else {
            using stride_t = Eigen::InnerStride<>;
            return Eigen::Map<vector_t, Eigen::Unaligned, stride_t>(vector.data(), vector.shape(0),
                                                                    stride_t(vector.stride(0)));
        }
    }
};

template <typename Matrix, int Rows, int Cols> struct EigenMapFactory<Matrix, Rows, Cols, 2u> {
    auto operator()(Matrix& matrix) {
        using real_t = std::remove_pointer_t<decltype(matrix.data())>;
        using matrix_t = copy_const<real_t, Eigen::Matrix<std::decay_t<real_t>, Rows, Cols>>;
        if constexpr (detail::traits<Matrix>::Packed) {
            return Eigen::Map<matrix_t>(matrix.data(), matrix.shape(0), matrix.shape(1));
        } else {
            using stride_t = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
            return Eigen::Map<matrix_t, Eigen::Unaligned, stride_t>(
                matrix.data(), matrix.shape(0), matrix.shape(1),
                stride_t(matrix.stride(1), matrix.stride(0)));
        }
    }
};

} // namespace detail

template <typename Tensor, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
auto EigenMap(Tensor& tensor) {
    detail::EigenMapFactory<Tensor, Rows, Cols, detail::traits<Tensor>::Dim> factory;
    return factory(tensor);
}

} // namespace tndm

#endif // EIGENMAP_20200609_H
