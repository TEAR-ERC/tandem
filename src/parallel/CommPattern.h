#ifndef COMMPATTERN_H
#define COMMPATTERN_H

#include <vector>
#include <mpi.h>

#include "mneme/displacements.hpp"
#include "MPITraits.h"

using mneme::Displacements;

namespace tndm {

class AllToAllV {
public:
    AllToAllV(std::vector<int>&& sndcnts, MPI_Comm comm = MPI_COMM_WORLD);
    AllToAllV(std::vector<int>&& sndcnts, std::vector<int>&& recvcnts, MPI_Comm comm = MPI_COMM_WORLD);
    AllToAllV(Displacements<int> const& sdispls, MPI_Comm comm = MPI_COMM_WORLD);

    void swap();

    auto const& getRecvcounts() const { return recvcounts; }
    auto const& getRDispls() const { return rdispls; }
    auto const& getSendcounts() const { return sendcounts; }
    auto const& getSDispls() const { return sdispls; }

    template <typename T>
    [[nodiscard]] std::vector<T> exchange(std::vector<T> const& dataToSend,
                                          MPI_Datatype const& mpiType = mpi_type_t<T>()) const {
        std::vector<T> recvdData(rdispls[procs-1] + recvcounts[procs-1]);
        MPI_Alltoallv(dataToSend.data(), sendcounts.data(), sdispls.data(), mpiType,
                      recvdData.data(), recvcounts.data(), rdispls.data(), mpiType, comm);
        return recvdData;
    }

private:
    std::vector<int> sendcounts;
    MPI_Comm comm;
    int procs;
    std::vector<int> recvcounts;
    Displacements<int> sdispls;
    Displacements<int> rdispls;
};

}

#endif // COMMPATTERN_H
