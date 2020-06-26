#ifndef ERROR_20200625_H
#define ERROR_20200625_H

#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "quadrules/AutoRule.h"
#include "tensor/EigenMap.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include <array>
#include <utility>

namespace tndm {

class SolutionInterface {
public:
    virtual ~SolutionInterface() {}

    virtual std::size_t numberOfQuantities() const = 0;
    /**
     * @brief Evaluate solution at quadrature points.
     *
     * @param coords Physical coordinate matrix of shape (D, numberOfPoints)
     * @param result Matrix with shape (numberOfPoints, numberOfQuantities)
     */
    virtual void operator()(Matrix<double> const& coords, Matrix<double>& result) const = 0;
};

/**
 * @brief Applies functional on coordinate matrix
 *
 * @tparam Func Must be a functional with operator(), e.g. a lambda, which takes
 * Vector<double> const& as input and returns a std::array.
 */
template <typename Func> class LambdaSolution : public SolutionInterface {
public:
    LambdaSolution(Func referenceFun) : func_(std::move(referenceFun)) {}

    std::size_t numberOfQuantities() const override {
        // Figure out the size of the array returned by func_
        return std::tuple_size_v<decltype(func_(std::declval<Vector<double> const&>()))>;
    }

    void operator()(Matrix<double> const& coords, Matrix<double>& result) const override {
        for (std::size_t q = 0; q < coords.shape(1); ++q) {
            std::size_t i = 0;
            for (auto&& r : func_(coords.subtensor(slice{}, q))) {
                assert(i < result.shape(1));
                result(q, i++) = r;
            }
        }
    }

private:
    Func func_;
};

template <std::size_t D> class Error {
public:
    /**
     * @brief Computes \sum_i ||numeric_i(x) - reference_i(x)||_2
     *
     * @param degree Polynomial degree of numeric solution
     * @param cl Curvilinear transformation
     * @param numeric Numeric solution tensor of shape
     *                (numberOfBasisFunctions, numberOfQuantities, numberOfElements)
     * @param reference Reference solution
     *
     * @return L2 error
     */
    static double L2(unsigned degree, Curvilinear<D>& cl, Tensor<const double, 3u> const& numeric,
                     SolutionInterface const& reference);
};

} // namespace tndm

#endif // ERROR_20200625_H
