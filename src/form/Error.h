#ifndef ERROR_20200625_H
#define ERROR_20200625_H

#include "mesh/LocalSimplexMesh.h"
#include "tensor/Tensor.h"

#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace tndm {

template <std::size_t D> class Curvilinear;
template <std::size_t D> class FiniteElementFunction;

class SolutionInterface {
public:
    virtual ~SolutionInterface() {}

    virtual std::size_t numQuantities() const = 0;
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

    std::size_t numQuantities() const override {
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
    static constexpr unsigned MinQuadratureOrder = 20;
    /**
     * @brief Computes \sum_i ||numeric_i(x) - reference_i(x)||_2
     *
     * @param cl Curvilinear transformation
     * @param numeric Finite element function
     * @param reference Reference solution
     * @param targetRank Error will be reduced on this rank
     * @param MPI_Comm MPI communicator
     *
     * @return L2 error
     */
    static double L2(Curvilinear<D>& cl, FiniteElementFunction<D> const& numeric,
                     SolutionInterface const& reference, int targetRank = 0,
                     MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Computes \sum_i ||numeric_i(x) - reference_i(x)||_2 on facets
     *
     * @param mesh Mesh
     * @param cl Curvilinear transformation
     * @param numeric Finite element function
     * @param reference Reference solution
     * @param fctNos Facets for which numeric is defined
     * @param targetRank Error will be reduced on this rank
     * @param MPI_Comm MPI communicator
     *
     * @return L2 error
     */
    static double L2(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl,
                     FiniteElementFunction<D - 1> const& numeric,
                     std::vector<std::size_t> const& fctNos, SolutionInterface const& reference,
                     int targetRank, MPI_Comm comm);

    /**
     * @brief Computes \sum_i \sum_j ||numeric_{i,j}(x) - reference_{i,j}(x)||_2
     *
     * Here, {.,j} denotes the derivative w.r.t. x_j.
     * Reference r is the Jacobian in row-major order, i.e.:
     * r_{1,1}, r_{1,2}, ..., r_{2, 1}, r_{2, 2}, ...
     *
     * @param cl Curvilinear transformation
     * @param numeric Finite element function
     * @param reference Reference solution
     *
     * @return Error measured in H1 semi-norm
     */
    static double H1_semi(Curvilinear<D>& cl, FiniteElementFunction<D> const& numeric,
                          SolutionInterface const& reference, int targetRank = 0,
                          MPI_Comm = MPI_COMM_WORLD);
};

} // namespace tndm

#endif // ERROR_20200625_H
