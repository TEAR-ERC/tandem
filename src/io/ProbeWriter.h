#ifndef PROBEWRITER_20210719_H
#define PROBEWRITER_20210719_H

#include "form/BoundaryMap.h"
#include "form/FiniteElementFunction.h"
#include "geometry/Curvilinear.h"
#include "io/Probe.h"
#include "io/TableWriter.h"
#include "mesh/LocalSimplexMesh.h"

#include <mneme/span.hpp>
#include <mpi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tndm {

template <std::size_t D> class ProbeWriter {
public:
    ProbeWriter(std::string_view prefix, std::unique_ptr<TableWriter> table_writer,
                std::vector<Probe<D>> const& probes, LocalSimplexMesh<D> const& mesh,
                std::shared_ptr<Curvilinear<D>> cl, MPI_Comm comm);

    void write(double time, mneme::span<FiniteElementFunction<D>> functions) const;

    auto begin() const { return elNos_.begin(); }
    auto end() const { return elNos_.end(); }
    std::vector<std::size_t> const& elNos() const { return elNos_; }

    auto num_probes() const { return probes_.size(); }

private:
    struct ProbeMeta {
        std::string name;
        std::string file_name;
        std::size_t no;
        std::array<double, D> xi;
        std::array<double, D> x;
    };

    void write_header(ProbeMeta const& p, mneme::span<FiniteElementFunction<D>> functions) const;

    std::unique_ptr<TableWriter> out_;
    std::vector<std::size_t> elNos_;
    std::vector<ProbeMeta> probes_;
};

} // namespace tndm

#endif // PROBEWRITER_20210719_H
