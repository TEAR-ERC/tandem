#include "CommPattern.h"

#include <cassert>
#include <cstddef>
#include <utility>

namespace tndm {

void make_displs(std::vector<int> const& counts, std::vector<int>& displs,
                 std::function<int(int)> permutation) {
    displs.resize(counts.size());
    displs[permutation(0)] = 0;
    for (std::size_t p = 0; p < counts.size() - 1; ++p) {
        displs[permutation(p + 1)] = displs[permutation(p)] + counts[permutation(p)];
    }
}

AllToAllV::AllToAllV(std::vector<int>&& sndcnts, MPI_Comm comm)
    : sendcounts(std::move(sndcnts)), comm(comm) {
    MPI_Comm_size(comm, &procs);
    assert(sendcounts.size() == procs);

    recvcounts.resize(procs);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    make_displs(sendcounts, sdispls);
    make_displs(recvcounts, rdispls);
}

AllToAllV::AllToAllV(std::vector<int>&& sndcnts, std::vector<int>&& recvcnts, MPI_Comm comm)
    : sendcounts(std::move(sndcnts)), comm(comm), recvcounts(std::move(recvcnts)) {
    MPI_Comm_size(comm, &procs);
    assert(sendcounts.size() == procs);
    assert(recvcounts.size() == procs);

    make_displs(sendcounts, sdispls);
    make_displs(recvcounts, rdispls);
}

void AllToAllV::swap() {
    sendcounts.swap(recvcounts);
    sdispls.swap(rdispls);
}

void AllToAllV::setRankPermutation(std::vector<int> const& permutation) {
    make_displs(sendcounts, sdispls, [&permutation](int p) { return permutation[p]; });
    make_displs(recvcounts, rdispls, [&permutation](int p) { return permutation[p]; });
}

GatherV::GatherV(int sendcount, int root, MPI_Comm comm)
    : sendcount_(sendcount), root_(root), comm_(comm) {
    int rank, size;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);
    if (rank == root_) {
        recvcounts_.resize(size);
    }
    MPI_Gather(&sendcount, 1, mpi_type_t<int>(), recvcounts_.data(), 1, mpi_type_t<int>(), root_,
               comm_);
    if (rank == root_) {
        make_displs(recvcounts_, displs_);
    }
}

} // namespace tndm
