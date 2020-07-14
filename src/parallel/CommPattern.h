#ifndef COMMPATTERN_H
#define COMMPATTERN_H

#include "MPITraits.h"
#include "util/Range.h"

#include <functional>
#include <mpi.h>
#include <numeric>
#include <vector>

namespace tndm {

class AllToAllV {
public:
    AllToAllV(std::vector<int>&& sndcnts, MPI_Comm comm = MPI_COMM_WORLD);
    AllToAllV(std::vector<int>&& sndcnts, std::vector<int>&& recvcnts,
              MPI_Comm comm = MPI_COMM_WORLD);

    void swap();
    void setRankPermutation(std::vector<int> const& permutation);

    auto recvRange(int p) const { return Range(rdispls[p], rdispls[p] + recvcounts[p]); }
    auto sendRange(int p) const { return Range(sdispls[p], sdispls[p] + sendcounts[p]); }

    template <typename T>
    [[nodiscard]] std::vector<T> exchange(std::vector<T> const& dataToSend,
                                          MPI_Datatype const& mpiType = mpi_type_t<T>()) const {
        std::vector<T> recvdData(std::accumulate(recvcounts.begin(), recvcounts.end(), 0));
        MPI_Alltoallv(dataToSend.data(), sendcounts.data(), sdispls.data(), mpiType,
                      recvdData.data(), recvcounts.data(), rdispls.data(), mpiType, comm);
        return recvdData;
    }

private:
    void makeDispls(
        std::vector<int> const& counts, std::vector<int>& displs,
        std::function<int(int)> permutation = [](int p) { return p; });

    std::vector<int> sendcounts;
    MPI_Comm comm;
    int procs;
    std::vector<int> recvcounts;
    std::vector<int> sdispls;
    std::vector<int> rdispls;
};

} // namespace tndm

#endif // COMMPATTERN_H
