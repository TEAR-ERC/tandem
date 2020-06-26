#include "Error.h"

namespace tndm {

template <std::size_t D>
double Error<D>::L2(RefElement<D> const& refElement, Curvilinear<D>& cl,
                    Tensor<const double, 3u> const& numeric, SolutionInterface const& reference) {
    auto rule = simplexQuadratureRule<D>(20);

    auto E = refElement.evaluateBasisAt(rule.points());
    Eigen::MatrixXd ET = EigenMap(E).transpose();
    Eigen::MatrixXd numAtQp(ET.rows(), numeric.shape(1));

    auto geoE = cl.evaluateBasisAt(rule.points());
    auto geoD_xi = cl.evaluateGradientAt(rule.points());
    auto J = Managed(cl.jacobianResultInfo(rule.size()));
    auto absDetJ = Managed(cl.detJResultInfo(rule.size()));
    auto coords = Managed(cl.mapResultInfo(rule.size()));

    auto ref = Managed<Matrix<double>>(rule.size(), reference.numberOfQuantities());

    double error = 0.0;
    for (std::size_t elNo = 0; elNo < cl.numElements(); ++elNo) {
        cl.jacobian(elNo, geoD_xi, J);
        cl.absDetJ(elNo, J, absDetJ);
        cl.map(elNo, geoE, coords);
        reference(coords, ref);
        auto numForEl = numeric.subtensor(slice{}, slice{}, elNo);
        numAtQp = ET * EigenMap(numForEl);
        // int (x - xref)^2 dV = w_q |J|_q (x_k E_{kq} - xref(x_q))^2
        double localError = 0;
        for (std::size_t j = 0; j < numAtQp.cols(); ++j) {
            for (std::size_t q = 0; q < numAtQp.rows(); ++q) {
                double e = ref(q, j) - numAtQp(q, j);
                localError += rule.weights()[q] * absDetJ(q) * e * e;
            }
        }
        error += localError;
    }

    return sqrt(error);
}

template class Error<2u>;
template class Error<3u>;

} // namespace tndm
