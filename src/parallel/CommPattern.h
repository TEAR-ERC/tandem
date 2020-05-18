#ifndef COMMPATTERN_H
#define COMMPATTERN_H

#include <vector>
#include <mpi.h>

#include "MPITraits.h"

namespace tndm {

class AllToAllV {
public:
    AllToAllV(std::vector<int>&& sndcnts, MPI_Comm comm = MPI_COMM_WORLD);

    void swap();

    std::vector<int> const& getRecvcounts() const { return recvcounts; }
    std::vector<int> const& getRDispls() const { return rdispls; }
    std::vector<int> const& getSendcounts() const { return sendcounts; }
    std::vector<int> const& getSDispls() const { return sdispls; }

    template<typename T>
    [[nodiscard]] std::vector<T> exchange(std::vector<T>& dataToSend, MPI_Datatype const& mpiType = mpi_type_t<T>()) const {
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
    std::vector<int> sdispls;
    std::vector<int> rdispls;
};

}

#endif // COMMPATTERN_H
