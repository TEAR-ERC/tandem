#include "CommPattern.h"
#include "util/Algorithm.h"

#include <cassert>

namespace tndm {

AllToAllV::AllToAllV(std::vector<int>&& sndcnts, MPI_Comm comm)
    : sendcounts(std::move(sndcnts)), comm(comm) {
    MPI_Comm_size(comm, &procs);
    assert(sendcounts.size() == procs);

    recvcounts.resize(procs);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    makeDispls(sendcounts, sdispls);
    makeDispls(recvcounts, rdispls);
}

AllToAllV::AllToAllV(std::vector<int>&& sndcnts, std::vector<int>&& recvcnts, MPI_Comm comm)
    : sendcounts(std::move(sndcnts)), comm(comm), recvcounts(std::move(recvcnts)) {
    MPI_Comm_size(comm, &procs);
    assert(sendcounts.size() == procs);
    assert(recvcounts.size() == procs);

    makeDispls(sendcounts, sdispls);
    makeDispls(recvcounts, rdispls);
}

void AllToAllV::makeDispls(std::vector<int> const& counts, std::vector<int>& displs,
                           std::function<int(int)> permutation) {
    displs.resize(counts.size());
    displs[permutation(0)] = 0;
    for (std::size_t p = 0; p < counts.size() - 1; ++p) {
        displs[permutation(p + 1)] = displs[permutation(p)] + counts[permutation(p)];
    }
}

void AllToAllV::swap() {
    sendcounts.swap(recvcounts);
    sdispls.swap(rdispls);
}

void AllToAllV::setRankPermutation(std::vector<int> const& permutation) {
    makeDispls(sendcounts, sdispls, [&permutation](int p) { return permutation[p]; });
    makeDispls(recvcounts, rdispls, [&permutation](int p) { return permutation[p]; });
}

} // namespace tndm
