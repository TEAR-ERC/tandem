#include "Error.h"
#include "form/FiniteElementFunction.h"

namespace tndm {

template <std::size_t D>
double Error<D>::L2(Curvilinear<D>& cl, FiniteElementFunction<D> const& numeric,
                    SolutionInterface const& reference, int targetRank, MPI_Comm comm) {
    auto rule = simplexQuadratureRule<D>(20);

    auto evalMatrix = numeric.evaluationMatrix(rule.points());
    auto numAtQp = Managed(numeric.mapResultInfo(rule.size()));

    auto geoE = cl.evaluateBasisAt(rule.points());
    auto geoD_xi = cl.evaluateGradientAt(rule.points());
    auto J = Managed(cl.jacobianResultInfo(rule.size()));
    auto absDetJ = Managed(cl.detJResultInfo(rule.size()));
    auto coords = Managed(cl.mapResultInfo(rule.size()));

    auto ref = Managed<Matrix<double>>(rule.size(), reference.numQuantities());

    double error = 0.0;
    for (std::size_t elNo = 0; elNo < numeric.numElements(); ++elNo) {
        cl.jacobian(elNo, geoD_xi, J);
        cl.absDetJ(elNo, J, absDetJ);
        cl.map(elNo, geoE, coords);
        reference(coords, ref);
        numeric.map(elNo, evalMatrix, numAtQp);
        // int (x - xref)^2 dV = w_q |J|_q (x_k E_{kq} - xref(x_q))^2
        double localError = 0;
        for (std::size_t j = 0; j < numeric.numQuantities(); ++j) {
            for (std::size_t q = 0; q < rule.size(); ++q) {
                double e = ref(q, j) - numAtQp(q, j);
                localError += rule.weights()[q] * absDetJ(q) * e * e;
            }
        }
        error += localError;
    }

    double globalError;
    MPI_Reduce(&error, &globalError, 1, mpi_type_t<double>(), MPI_SUM, targetRank, comm);

    return sqrt(globalError);
}

template class Error<2u>;
template class Error<3u>;

} // namespace tndm
