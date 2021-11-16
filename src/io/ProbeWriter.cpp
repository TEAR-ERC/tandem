#include "ProbeWriter.h"

#include "geometry/PointLocator.h"
#include "io/ProbeWriterUtil.h"

#include <iomanip>
#include <ios>
#include <mpi.h>
#include <unordered_map>
#include <unordered_set>

namespace tndm {

template <std::size_t D>
ProbeWriter<D>::ProbeWriter(std::string_view prefix, std::vector<Probe<D>> const& probes,
                            LocalSimplexMesh<D> const& mesh, std::shared_ptr<Curvilinear<D>> cl,
                            MPI_Comm comm) {
    auto pl = PointLocator<D>(cl);
    auto range = Range<std::size_t>(0, mesh.elements().localSize());

    std::vector<std::pair<std::size_t, PointLocatorResult<D>>> located_probes;
    located_probes.reserve(probes.size());
    for (std::size_t p = 0; p < probes.size(); ++p) {
        auto result = pl.locate(probes[p].x, range.begin(), range.end());
        located_probes.emplace_back(std::make_pair(p, result));
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    clean_duplicate_probes(probes, located_probes, comm);

    std::unordered_set<std::size_t> myElNos;
    for (auto const& p : located_probes) {
        myElNos.emplace(p.second.no);
    }

    elNos_.reserve(myElNos.size());
    std::unordered_map<std::size_t, std::size_t> elNo2OutNo;

    std::size_t outNo = 0;
    for (auto const& elNo : myElNos) {
        elNo2OutNo[elNo] = outNo++;
        elNos_.emplace_back(elNo);
    }

    probes_.reserve(located_probes.size());
    for (auto const& p : located_probes) {
        auto file_name = std::string(prefix);
        file_name += probes[p.first].name + ".dat";
        probes_.emplace_back(ProbeMeta{probes[p.first].name, file_name, elNo2OutNo[p.second.no],
                                       p.second.xi, p.second.x});
    }
}

template <std::size_t D>
void ProbeWriter<D>::write_header(std::ofstream& file, ProbeMeta const& p,
                                  mneme::span<FiniteElementFunction<D>> functions) const {
    file << "# TITLE = \"Station " << p.name << " (x = [";
    for (std::size_t d = 0; d < D; ++d) {
        file << p.x[d] << ", ";
    }
    file.seekp(-2, std::ios::cur);
    file << "])\"" << std::endl;
    file << "# VARIABLES = \"Time\"";
    for (auto const& function : functions) {
        for (std::size_t q = 0; q < function.numQuantities(); ++q) {
            file << ",\"" << function.name(q) << "\"";
        }
    }
    file << std::endl;
}

template <std::size_t D>
void ProbeWriter<D>::write(double time, mneme::span<FiniteElementFunction<D>> functions) const {
    for (auto const& probe : probes_) {
        std::ofstream file;
        if (time <= 0.0) {
            file.open(probe.file_name, std::ios::out);
            write_header(file, probe, functions);
        } else {
            file.open(probe.file_name, std::ios::app);
        }

        file << std::scientific << std::setprecision(15);
        file << time;
        for (auto const& function : functions) {
            auto result = Managed<Matrix<double>>(function.mapResultInfo(1));
            auto E = function.evaluationMatrix({probe.xi});
            function.map(probe.no, E, result);
            for (std::size_t p = 0; p < function.numQuantities(); ++p) {
                file << " " << result(0, p);
            }
        }
        file << std::endl;
        file.close();
    }
}

template class ProbeWriter<2u>;
template class ProbeWriter<3u>;

} // namespace tndm
