#ifndef BC_20200910_H
#define BC_20200910_H

#include <mpi.h>

namespace tndm {

enum class BC : int { None = 0, Natural = 1, Fault = 3, Dirichlet = 5 };

// See form/MPITraits.h :
template <typename T> struct mpi_type;

template <> struct mpi_type<BC> {
    static MPI_Datatype type() { return MPI_INT; }
};


}

#endif // BC_20200910_H
