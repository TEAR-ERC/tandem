#ifndef SORTEDDISTRIBUTION_20200710_H
#define SORTEDDISTRIBUTION_20200710_H

#include <cassert>
#include <cstddef>
#include <mpi.h>
#include <vector>

namespace tndm {

std::vector<std::size_t> makeSortedDistribution(std::size_t num, MPI_Comm comm = MPI_COMM_WORLD);

class SortedDistributionToRank {
public:
    SortedDistributionToRank(std::vector<std::size_t> const& dist)
        : dist(dist), guessSize(1 + (dist.back() - 1) / (dist.size() - 1)) {}

    int operator()(std::size_t id) const;

private:
    std::vector<std::size_t> const& dist;
    std::size_t guessSize;
};

} // namespace tndm

#endif // SORTEDDISTRIBUTION_20200710_H
