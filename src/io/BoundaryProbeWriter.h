#ifndef BOUNDARYPROBEWRITER_20210419_H
#define BOUNDARYPROBEWRITER_20210419_H

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

template <std::size_t D> class BoundaryProbeWriter {
public:
    BoundaryProbeWriter(std::string_view prefix, std::unique_ptr<TableWriter> table_writer,
                        std::vector<Probe<D>> const& probes, LocalSimplexMesh<D> const& mesh,
                        std::shared_ptr<Curvilinear<D>> cl, BoundaryMap const& bnd_map,
                        MPI_Comm comm);

    void write(double time, mneme::span<FiniteElementFunction<D - 1>> functions) const;

    auto begin() const { return bndNos_.begin(); }
    auto end() const { return bndNos_.end(); }
    auto num_probes() const { return probes_.size(); }
    std::vector<std::size_t> const& bndNos() const { return bndNos_; }
    void truncate_after_restart(std::size_t p);

private:
    struct ProbeMeta {
        std::string name;
        std::string file_name;
        std::size_t no;
        std::array<double, D - 1> chi;
        std::array<double, D> x;
    };

    void write_header(ProbeMeta const& p,
                      mneme::span<FiniteElementFunction<D - 1>> functions) const;

    std::unique_ptr<TableWriter> out_;
    std::vector<std::size_t> bndNos_;
    std::vector<ProbeMeta> probes_;
};

} // namespace tndm

#endif // BOUNDARYPROBEWRITER_20210419_H
