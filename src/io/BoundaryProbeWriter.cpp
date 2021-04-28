#include "BoundaryProbeWriter.h"

#include "geometry/PointLocator.h"

#include <cassert>
#include <iomanip>
#include <ios>
#include <mpi.h>
#include <unordered_map>
#include <unordered_set>

namespace tndm {

template <std::size_t D>
BoundaryProbeWriter<D>::BoundaryProbeWriter(std::string_view prefix,
                                            std::vector<Probe<D>> const& probes,
                                            LocalSimplexMesh<D> const& mesh,
                                            std::shared_ptr<Curvilinear<D>> cl,
                                            BoundaryMap const& bnd_map, MPI_Comm comm) {
    auto bpl =
        BoundaryPointLocator<D>(std::make_shared<PointLocator<D>>(cl), mesh, bnd_map.fctNos());

    std::vector<std::pair<std::size_t, BoundaryPointLocatorResult<D>>> located_probes;
    located_probes.reserve(probes.size());
    std::vector<double> min_dist;
    min_dist.reserve(probes.size());
    for (std::size_t p = 0; p < probes.size(); ++p) {
        auto result = bpl.locate(probes[p].x);
        located_probes.emplace_back(std::make_pair(p, result));
        min_dist.emplace_back(result.dist);
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    // The probe might be found on several ranks, thus we take the rank with minimum distance.
    MPI_Allreduce(MPI_IN_PLACE, min_dist.data(), min_dist.size(), MPI_DOUBLE, MPI_MIN, comm);

    /* There might still be a tie, i.e. multiple ranks find a closest point with the same minimum
     * distance (in particular if ghost layers overlap). Therefore, we arbitrarily assign
     * the probe to the rank with lower rank number.
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
            ss << "Could not closest face for probe at ";
            for (auto x : probes[i].x) {
                ss << x << " ";
            }
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

    std::unordered_set<std::size_t> myFctNos;
    for (auto const& p : located_probes) {
        myFctNos.emplace(p.second.no);
    }

    bndNos_.reserve(myFctNos.size());
    std::unordered_map<std::size_t, std::size_t> fctNo2OutNo;

    std::size_t outNo = 0;
    for (auto const& fctNo : myFctNos) {
        fctNo2OutNo[fctNo] = outNo++;
        bndNos_.emplace_back(bnd_map.bndNo(fctNo));
    }

    probes_.reserve(located_probes.size());
    for (auto const& p : located_probes) {
        auto file_name = std::string(prefix);
        file_name += probes[p.first].name + ".dat";
        probes_.emplace_back(ProbeMeta{probes[p.first].name, file_name, fctNo2OutNo[p.second.no],
                                       p.second.chi, p.second.x});
    }
}

template <std::size_t D>
void BoundaryProbeWriter<D>::write_header(std::ofstream& file, ProbeMeta const& p,
                                          FiniteElementFunction<D - 1> const& function) const {
    file << "TITLE = \"Station " << p.name << " (x = [";
    for (std::size_t d = 0; d < D; ++d) {
        file << p.x[d] << ", ";
    }
    file.seekp(-2, std::ios::cur);
    file << "])\"" << std::endl;
    file << "VARIABLES = \"Time\"";
    for (std::size_t q = 0; q < function.numQuantities(); ++q) {
        file << ",\"" << function.name(q) << "\"";
    }
    file << std::endl;
}

template <std::size_t D>
void BoundaryProbeWriter<D>::write(double time,
                                   FiniteElementFunction<D - 1> const& function) const {
    assert(function.numElements() == probes_.size());

    auto result = Managed<Matrix<double>>(function.mapResultInfo(1));
    for (auto const& probe : probes_) {
        std::ofstream file;
        if (time <= 0.0) {
            file.open(probe.file_name, std::ios::out);
            write_header(file, probe, function);
        } else {
            file.open(probe.file_name, std::ios::app);
        }

        auto E = function.evaluationMatrix({probe.chi});
        function.map(probe.no, E, result);
        file << std::scientific << std::setprecision(15);
        file << time;
        for (std::size_t p = 0; p < function.numQuantities(); ++p) {
            file << " " << result(0, p);
        }
        file << std::endl;
        file.close();
    }
}

template class BoundaryProbeWriter<2u>;
template class BoundaryProbeWriter<3u>;

} // namespace tndm
