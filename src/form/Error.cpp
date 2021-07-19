#include "Error.h"
#include "form/FiniteElementFunction.h"
#include "geometry/Curvilinear.h"
#include "parallel/MPITraits.h"
#include "quadrules/AutoRule.h"
#include "tensor/Managed.h"

#include <cmath>

namespace tndm {

template <std::size_t D>
double Error<D>::L2(Curvilinear<D>& cl, FiniteElementFunction<D> const& numeric,
                    SolutionInterface const& reference, int targetRank, MPI_Comm comm) {
    auto rule = simplexQuadratureRule<D>(MinQuadratureOrder);

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

template <std::size_t D>
double Error<D>::L2(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl,
                    FiniteElementFunction<D - 1> const& numeric,
                    std::vector<std::size_t> const& fctNos, SolutionInterface const& reference,
                    int targetRank, MPI_Comm comm) {
    auto rule = simplexQuadratureRule<D - 1>(MinQuadratureOrder);

    auto evalMatrix = numeric.evaluationMatrix(rule.points());
    auto numAtQp = Managed(numeric.mapResultInfo(rule.size()));

    std::vector<Managed<Matrix<double>>> geoE;
    std::vector<Managed<Tensor<double, 3u>>> geoD_xi;

    for (std::size_t f = 0; f < D + 1u; ++f) {
        auto facetParam = cl.facetParam(f, rule.points());
        geoE.emplace_back(cl.evaluateBasisAt(facetParam));
        geoD_xi.emplace_back(cl.evaluateGradientAt(facetParam));
    }
    auto J = Managed(cl.jacobianResultInfo(rule.size()));
    auto Jinv = Managed(cl.jacobianResultInfo(rule.size()));
    auto detJ = Managed(cl.detJResultInfo(rule.size()));
    auto normal = Managed(cl.normalResultInfo(rule.size()));
    auto coords = Managed(cl.mapResultInfo(rule.size()));

    auto ref = Managed<Matrix<double>>(rule.size(), reference.numQuantities());

    double error = 0.0;
    for (std::size_t bndNo = 0; bndNo < numeric.numElements(); ++bndNo) {
        auto fctNo = fctNos[bndNo];
        auto elNos = mesh.template upward<D - 1u>(fctNo);
        assert(elNos.size() >= 1u);
        auto elNo = elNos[0];
        assert(elNo < cl.numElements());
        auto dws = mesh.template downward<D - 1u, D>(elNo);
        int localFaceNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));
        assert(localFaceNo < D + 1u);

        cl.jacobian(elNo, geoD_xi[localFaceNo], J);
        cl.jacobianInv(J, Jinv);
        cl.detJ(elNo, J, detJ);
        cl.normal(localFaceNo, detJ, Jinv, normal);

        cl.map(elNo, geoE[localFaceNo], coords);
        reference(coords, ref);
        numeric.map(bndNo, evalMatrix, numAtQp);
        // int (x - xref)^2 dV = w_q ||n||_q (x_k E_{kq} - xref(x_q))^2
        double localError = 0;
        for (std::size_t q = 0; q < rule.size(); ++q) {
            double nl = 0.0;
            for (std::size_t d = 0; d < D; ++d) {
                auto n_d = normal(d, q);
                nl += n_d * n_d;
            }
            nl = sqrt(nl);
            for (std::size_t j = 0; j < numeric.numQuantities(); ++j) {
                double e = ref(q, j) - numAtQp(q, j);
                localError += rule.weights()[q] * nl * e * e;
            }
        }
        error += localError;
    }

    double globalError;
    MPI_Reduce(&error, &globalError, 1, mpi_type_t<double>(), MPI_SUM, targetRank, comm);

    return sqrt(globalError);
}

template <std::size_t D>
double Error<D>::H1_semi(Curvilinear<D>& cl, FiniteElementFunction<D> const& numeric,
                         SolutionInterface const& reference, int targetRank, MPI_Comm comm) {
    auto rule = simplexQuadratureRule<D>(MinQuadratureOrder);

    auto evalTensor = numeric.gradientEvaluationTensor(rule.points());
    auto gradAtQp = Managed(numeric.gradientResultInfo(rule.size()));

    auto geoE = cl.evaluateBasisAt(rule.points());
    auto geoD_xi = cl.evaluateGradientAt(rule.points());
    auto J = Managed(cl.jacobianResultInfo(rule.size()));
    auto Jinv = Managed(cl.jacobianResultInfo(rule.size()));
    auto absDetJ = Managed(cl.detJResultInfo(rule.size()));
    auto coords = Managed(cl.mapResultInfo(rule.size()));

    assert(reference.numQuantities() == numeric.numQuantities() * D);
    auto ref = Managed<Matrix<double>>(rule.size(), reference.numQuantities());

    auto Q = numeric.numQuantities();
    double error = 0.0;
    for (std::size_t elNo = 0; elNo < numeric.numElements(); ++elNo) {
        cl.jacobian(elNo, geoD_xi, J);
        cl.jacobianInv(J, Jinv);
        cl.absDetJ(elNo, J, absDetJ);
        cl.map(elNo, geoE, coords);
        reference(coords, ref);
        numeric.gradient(elNo, evalTensor, Jinv, gradAtQp);
        double localError = 0;
        for (std::size_t i = 0; i < Q; ++i) {
            for (std::size_t j = 0; j < D; ++j) {
                for (std::size_t q = 0; q < rule.size(); ++q) {
                    double e = ref(q, i * Q + j) - gradAtQp(q, i, j);
                    localError += rule.weights()[q] * absDetJ(q) * e * e;
                }
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
