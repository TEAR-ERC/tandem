#include "BoundaryProbeWriter.h"

#include "geometry/PointLocator.h"
#include "io/ProbeWriterUtil.h"
#include "util/LinearAllocator.h"

#ifdef EXPERIMENTAL_FS
#include <experimental/filesystem>
#else
#include <filesystem>
#endif

#include <mpi.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#ifdef EXPERIMENTAL_FS
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

namespace tndm {

template <std::size_t D>
BoundaryProbeWriter<D>::BoundaryProbeWriter(std::string_view prefix,
                                            std::unique_ptr<TableWriter> table_writer,
                                            std::vector<Probe<D>> const& probes,
                                            LocalSimplexMesh<D> const& mesh,
                                            std::shared_ptr<Curvilinear<D>> cl,
                                            BoundaryMap const& bnd_map, MPI_Comm comm)
    : out_(std::move(table_writer)) {
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
        file_name += probes[p.first].name;
        file_name += out_->default_extension();
        probes_.emplace_back(ProbeMeta{probes[p.first].name, file_name, fctNo2OutNo[p.second.no],
                                       p.second.chi, p.second.x});
    }
}

template <std::size_t D>
void BoundaryProbeWriter<D>::write_header(
    ProbeMeta const& p, mneme::span<FiniteElementFunction<D - 1>> functions) const {
    std::stringstream s;
    s << "Station " << p.name << " (x = [";
    for (std::size_t d = 0; d < D; ++d) {
        s << p.x[d] << ", ";
    }
    s.seekp(-2, std::ios::cur);
    s << "])";
    out_->add_title(s.str());
    *out_ << beginheader << "Time";
    for (auto const& function : functions) {
        for (std::size_t q = 0; q < function.numQuantities(); ++q) {
            *out_ << function.name(q);
        }
    }
    *out_ << endheader;
}

template <std::size_t D>
void BoundaryProbeWriter<D>::write(double time,
                                   mneme::span<FiniteElementFunction<D - 1>> functions) const {

    for (auto const& probe : probes_) {
        if (time <= 0.0) {
            out_->open(probe.file_name, false);
            write_header(probe, functions);
        } else {
            if (fs::exists(probe.file_name)) {
                out_->open(probe.file_name, true);
            } else {
                out_->open(probe.file_name, false);
                write_header(probe, functions);
            }
        }

        *out_ << time;
        for (auto const& function : functions) {
            auto result = Managed<Matrix<double>>(function.mapResultInfo(1));
            auto E = function.evaluationMatrix({probe.chi});
            function.map(probe.no, E, result);
            for (std::size_t p = 0; p < function.numQuantities(); ++p) {
                *out_ << result(0, p);
            }
        }
        *out_ << endrow;
        out_->close();
    }
}

template class BoundaryProbeWriter<2u>;
template class BoundaryProbeWriter<3u>;

} // namespace tndm
