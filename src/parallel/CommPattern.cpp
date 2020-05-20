#include "CommPattern.h"

#include <cassert>

namespace tndm {

AllToAllV::AllToAllV(std::vector<int>&& sndcnts, MPI_Comm comm)
    : sendcounts(std::move(sndcnts)), comm(comm) {
    MPI_Comm_size(comm, &procs);
    assert(sendcounts.size() == procs);

    recvcounts.resize(procs);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    sdispls.make(sendcounts);
    rdispls.make(recvcounts);
}

AllToAllV::AllToAllV(std::vector<int>&& sndcnts, std::vector<int>&& recvcnts, MPI_Comm comm)
    : sendcounts(std::move(sndcnts)), comm(comm), recvcounts(std::move(recvcnts)) {
    MPI_Comm_size(comm, &procs);
    assert(sendcounts.size() == procs);
    assert(recvcounts.size() == procs);

    sdispls.make(sendcounts);
    rdispls.make(recvcounts);
}

AllToAllV::AllToAllV(Displacements<int> const& sdispls, MPI_Comm comm)
    : sdispls(sdispls), comm(comm) {
    MPI_Comm_size(comm, &procs);
    sendcounts.resize(procs);
    for (int p = 0; p < procs; ++p) {
        sendcounts[p] = sdispls.count(p);
    }

    recvcounts.resize(procs);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    rdispls.make(recvcounts);
}

void AllToAllV::swap() {
    sendcounts.swap(recvcounts);
    sdispls.swap(rdispls);
}

}
