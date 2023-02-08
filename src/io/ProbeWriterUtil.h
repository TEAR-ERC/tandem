#ifndef PROBEWRITERUTIL_20210719_H
#define PROBEWRITERUTIL_20210719_H

#include "io/Probe.h"

#include <mpi.h>
#include <sstream>
#include <vector>

namespace tndm {

template <std::size_t D, template <std::size_t> typename ResultT>
void clean_duplicate_probes(std::vector<Probe<D>> const& probes,
                            std::vector<std::pair<std::size_t, ResultT<D>>>& located_probes,
                            MPI_Comm comm) {
    std::vector<double> min_dist;
    min_dist.reserve(located_probes.size());
    for (auto const& [probe_id, result] : located_probes) {
        min_dist.emplace_back(result.dist);
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    // The probe might be found on several ranks, thus we take the rank with minimum distance.
    MPI_Allreduce(MPI_IN_PLACE, min_dist.data(), min_dist.size(), MPI_DOUBLE, MPI_MIN, comm);

    /* There might still be a tie, i.e. multiple ranks find a closest point with the same minimum
     * distance. Therefore, we arbitrarily assign the probe to the rank with lower rank number.
     */
    auto min_rank = std::vector<int>(min_dist.size());
    for (auto const& p : located_probes) {
        min_rank[p.first] =
            p.second.dist == min_dist[p.first] ? rank : std::numeric_limits<int>::max();
    }

    MPI_Allreduce(MPI_IN_PLACE, min_rank.data(), min_rank.size(), MPI_INT, MPI_MIN, comm);
    for (std::size_t i = 0; i < min_rank.size(); ++i) {
        if (min_dist[i] == std::numeric_limits<double>::max() ||
            min_rank[i] == std::numeric_limits<int>::max()) {
            std::stringstream ss;
            ss << "Could not closest face [sic -- cpranger] for probe at ";
            for (auto x : probes[i].x) {
                ss << x << " ";
            }
			ss << "-- min_dist[i] = " << min_dist[i] << ",  min_rank[i] = " << min_rank[i];
            throw std::runtime_error(ss.str());
        }
    }

    for (auto it = located_probes.begin(); it != located_probes.end();) {
        if (rank != min_rank[it->first]) {
            it = located_probes.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace tndm

#endif // PROBEWRITERUTIL_20210719_H
