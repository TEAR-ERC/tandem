#ifndef DISTRIBUTE_H
#define DISTRIBUTE_H

#include <mpi.h>
#include "util/Range.h"

namespace tndm {
    template<typename IntT>
    Range<IntT> distribute(Range<IntT> const& range, int rank, int size) {
        IntT perRank = range.length()/size;
        IntT extras = range.length();
        if (perRank != 0) {
            extras %= perRank;
        }

        auto from = range.from + rank * perRank + (rank < extras ? rank : extras);
        return Range(from,
                     from + perRank + (rank < extras ? 1 : 0));
    }

    template <typename IntT>
    auto distribute(Range<IntT> const& range, MPI_Comm comm = MPI_COMM_WORLD) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        return distribute<IntT>(range, rank, size);
    }
}

#endif // DISTRIBUTE_H
