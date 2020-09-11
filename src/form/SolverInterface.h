#ifndef SOLVERINTERFACE_20200910_H
#define SOLVERINTERFACE_20200910_H

#include "tensor/Tensor.h"
#include <mpi.h>

#include <cstddef>

namespace tndm {

class SolverInterface {
public:
    void create_mat(std::size_t blockSize, std::size_t numLocalElems, unsigned const* numLocal,
                    unsigned const* numGhost, bool reuse, MPI_Comm comm) {}
    void create_vec(std::size_t blockSize, std::size_t numLocalElems, bool reuse, MPI_Comm comm) {}

    void begin_assembly() {}
    void add_block(std::size_t ib, std::size_t jb, Matrix<double> const& A) {}
    void end_assembly() {}

    void begin_rhs() {}
    void add_rhs(std::size_t ib, Vector<double> const& B) {}
    void end_rhs() {}
};

} // namespace tndm

#endif // SOLVERINTERFACE_20200910_H
