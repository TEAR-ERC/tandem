#include "BoundaryProbeWriter.h"

#include "geometry/PointLocator.h"
#include "io/ProbeWriterUtil.h"
#include "util/LinearAllocator.h"

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
        BoundaryPointLocator<D>(std::make_shared<PointLocator<D>>(cl), mesh, bnd_map.localFctNos());

    std::vector<std::pair<std::size_t, BoundaryPointLocatorResult<D>>> located_probes;
    located_probes.reserve(probes.size());
    for (std::size_t p = 0; p < probes.size(); ++p) {
        auto result = bpl.locate(probes[p].x);
        located_probes.emplace_back(std::make_pair(p, result));
    }

    clean_duplicate_probes(probes, located_probes, comm);

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
void BoundaryProbeWriter<D>::write_header(
    std::ofstream& file, ProbeMeta const& p,
    mneme::span<FiniteElementFunction<D - 1>> functions) const {
    file << "TITLE = \"Station " << p.name << " (x = [";
    for (std::size_t d = 0; d < D; ++d) {
        file << p.x[d] << ", ";
    }
    file.seekp(-2, std::ios::cur);
    file << "])\"" << std::endl;
    file << "VARIABLES = \"Time\"";
    for (auto const& function : functions) {
        for (std::size_t q = 0; q < function.numQuantities(); ++q) {
            file << ",\"" << function.name(q) << "\"";
        }
    }
    file << std::endl;
}

template <std::size_t D>
void BoundaryProbeWriter<D>::write(double time,
                                   mneme::span<FiniteElementFunction<D - 1>> functions) const {
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
            auto E = function.evaluationMatrix({probe.chi});
            function.map(probe.no, E, result);
            for (std::size_t p = 0; p < function.numQuantities(); ++p) {
                file << " " << result(0, p);
            }
        }
        file << std::endl;
        file.close();
    }
}

template class BoundaryProbeWriter<2u>;
template class BoundaryProbeWriter<3u>;

} // namespace tndm
