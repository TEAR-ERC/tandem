#include "CommPattern.h"

#include <cassert>

namespace tndm {

AllToAllV::AllToAllV(std::vector<int>&& sndcnts, MPI_Comm comm)
    : sendcounts(std::move(sndcnts)), comm(comm) {
    MPI_Comm_size(comm, &procs);
    assert(sendcounts.size() == procs);

    recvcounts.resize(procs);
    sdispls.resize(procs);
    rdispls.resize(procs);

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int p = 1; p < procs+1; ++p) {
        sdispls[p] = sdispls[p-1] + sendcounts[p-1];
        rdispls[p] = rdispls[p-1] + recvcounts[p-1];
    }
}

void AllToAllV::swap() {
    sendcounts.swap(recvcounts);
    sdispls.swap(rdispls);
}

}
