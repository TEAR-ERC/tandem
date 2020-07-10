#include "SortedDistribution.h"

#include "MPITraits.h"

namespace tndm {

std::vector<std::size_t> makeSortedDistribution(std::size_t num, MPI_Comm comm) {
    int procs;
    MPI_Comm_size(comm, &procs);

    std::vector<std::size_t> dist(procs + 1);
    dist[0] = 0;
    MPI_Allgather(&num, 1, mpi_type_t<std::size_t>(), dist.data() + 1, 1, mpi_type_t<std::size_t>(),
                  comm);
    for (int p = 1; p < procs + 1; ++p) {
        dist[p] += dist[p - 1];
    }
    return dist;
}

int SortedDistributionToRank::operator()(std::size_t id) const {
    int guess = static_cast<int>(id / guessSize);
    auto procs = dist.size() - 1;
    while (guess > 0 && id < dist[guess]) {
        --guess;
    }
    while (guess < procs && id >= dist[guess + 1]) {
        ++guess;
    }
    return guess;
}
} // namespace tndm
