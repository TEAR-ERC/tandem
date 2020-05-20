#ifndef SORTEDDISTRIBUTION_H
#define SORTEDDISTRIBUTION_H

#include <cassert>
#include <cstddef>
#include <vector>
#include <mpi.h>

namespace tndm {

std::vector<std::size_t> makeSortedDistribution(std::size_t num, MPI_Comm comm = MPI_COMM_WORLD);

class SortedDistributionToRank {
public:
    SortedDistributionToRank(std::vector<std::size_t> const& dist) : dist(dist), guessSize(dist.back() / (dist.size()-1)) {}

    int operator()(std::size_t id) const;

private:
    std::vector<std::size_t> const& dist;
    std::size_t guessSize;
};

}


#endif // SORTEDDISTRIBUTION_H
