#include "ProbeWriter.h"

#include "geometry/PointLocator.h"
#include "io/ProbeWriterUtil.h"

#include <filesystem>
#include <fstream>
#include <mpi.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace tndm {

template <std::size_t D>
ProbeWriter<D>::ProbeWriter(std::string_view prefix, std::unique_ptr<TableWriter> table_writer,
                            std::vector<Probe<D>> const& probes, LocalSimplexMesh<D> const& mesh,
                            std::shared_ptr<Curvilinear<D>> cl, MPI_Comm comm)
    : out_(std::move(table_writer)) {
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
        file_name += probes[p.first].name;
        file_name += out_->default_extension();
        probes_.emplace_back(ProbeMeta{probes[p.first].name, file_name, elNo2OutNo[p.second.no],
                                       p.second.xi, p.second.x});
    }
}

template <std::size_t D>
void ProbeWriter<D>::write_header(ProbeMeta const& p,
                                  mneme::span<FiniteElementFunction<D>> functions) const {
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

template <std::size_t D> void ProbeWriter<D>::truncate_after_restart(std::size_t p) {
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
void ProbeWriter<D>::write(double time, mneme::span<FiniteElementFunction<D>> functions) const {
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
            auto E = function.evaluationMatrix({probe.xi});
            function.map(probe.no, E, result);
            for (std::size_t p = 0; p < function.numQuantities(); ++p) {
                *out_ << result(0, p);
            }
        }
        *out_ << endrow;
        out_->close();
    }
}

template class ProbeWriter<2u>;
template class ProbeWriter<3u>;

} // namespace tndm
