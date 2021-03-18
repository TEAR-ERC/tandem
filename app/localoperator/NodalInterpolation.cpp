#include "NodalInterpolation.h"
#include "util/LinearAllocator.h"

#include <cassert>

namespace tndm {

template <std::size_t D>
void NodalInterpolation<D>::assemble(unsigned to_degree, unsigned from_degree, Matrix<double>& I,
                                     LinearAllocator<double>& scratch) const {

    assert(to_degree > from_degree);
    assert(to_degree < spaces_.size());

    auto const& V_to = spaces_[to_degree].vandermonde();
    auto const& VInv_from = spaces_[from_degree].vandermondeInv();

    std::size_t N1 = V_to.rows();
    std::size_t N2 = VInv_from.cols();

    assert(I.shape(0) == N1 * numQuantities);
    assert(I.shape(1) == N2 * numQuantities);

    double* mem = scratch.allocate(N1 * N2);
    auto tmp = Eigen::Map<Eigen::MatrixXd>(mem, N1, N2);

    tmp = V_to.leftCols(N2) * VInv_from;

    I.set_zero();
    for (int l = 0; l < N2; ++l) {
        for (int p = 0; p < numQuantities_; ++p) {
            for (int k = 0; k < N1; ++k) {
                I(k + p * N1, l + p * N2) = tmp(k, l);
            }
        }
    }
}

template class NodalInterpolation<1ul>;
template class NodalInterpolation<2ul>;
template class NodalInterpolation<3ul>;

} // namespace tndm
