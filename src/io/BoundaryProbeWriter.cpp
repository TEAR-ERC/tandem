#include "BoundaryProbeWriter.h"

#include "geometry/PointLocator.h"
#include "io/ProbeWriterUtil.h"
#include "util/LinearAllocator.h"

#include <filesystem>
#include <fstream>
#include <mpi.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

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

template <std::size_t D> void BoundaryProbeWriter<D>::truncate_after_restart(std::size_t p) {
    for (auto const& probe : probes_) {
        // Open file in read mode to read content
        std::ifstream file(probe.file_name);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + probe.file_name +
                                     " for truncation (checkpoint restart)");
            return;
        }

        std::stringstream buffer;
        std::string line;
        int line_num = 0;

        // Read file line by line into a stringstream, up to line p + 2
        while (std::getline(file, line)) {
            if (line_num >= p + 2) {
                break; // Stop reading after reaching the target line
            }
            buffer << line << '\n';
            line_num++;
        }
        file.close();

        // Check if enough lines were read
        if (line_num < p + 2) {
            std::cerr << p + 2 << " lines expected in " << probe.file_name
                      << " when truncating (checkpoint restart), but read only " << line_num
                      << "\n";
        }

        // Reopen the original file in write mode to overwrite its content
        std::ofstream out_file(probe.file_name, std::ios::trunc);
        if (!out_file.is_open()) {
            throw std::runtime_error("Failed to open file: " + probe.file_name +
                                     " for writing truncated content");
            return;
        }

        // Write the truncated content back
        out_file << buffer.str();
        out_file.close();

        std::cout << "Done truncating " << probe.file_name << " to " << p << " records.\n";
    }
}

template <std::size_t D>
void BoundaryProbeWriter<D>::write(double time,
                                   mneme::span<FiniteElementFunction<D - 1>> functions) const {
    for (auto const& probe : probes_) {
        if (time <= 0.0) {
            out_->open(probe.file_name, false);
            write_header(probe, functions);
        } else {
            if (std::filesystem::exists(probe.file_name)) {
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
