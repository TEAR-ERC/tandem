#ifndef SEASWRITER_20201006_H
#define SEASWRITER_20201006_H

#include "tandem/AdaptiveOutputStrategy.h"

#include "geometry/Curvilinear.h"
#include "interface/BlockVector.h"
#include "io/BoundaryProbeWriter.h"
#include "io/PVDWriter.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/LocalSimplexMesh.h"

#include <mpi.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tndm {

class SeasWriter {
public:
    SeasWriter(std::string_view prefix, AdaptiveOutputInterval oi) : prefix_(prefix), oi_(oi) {}
    virtual ~SeasWriter() {}

    void monitor(double time, BlockVector const& state, double VMax) {
        double delta_time = time - last_output_time_;
        if (oi_(delta_time, last_output_VMax_, VMax)) {
            write_step(time, state);
            ++output_step_;
            last_output_time_ = time;
            last_output_VMax_ = VMax;
        }
    }

protected:
    virtual void write_step(double time, BlockVector const& state) = 0;

    std::string name() const {
        std::stringstream ss;
        ss << prefix_ << "_" << output_step_;
        return ss.str();
    }

    std::string prefix_;
    AdaptiveOutputInterval oi_;

    PVDWriter pvd_;
    std::size_t output_step_ = 0;
    double last_output_time_ = std::numeric_limits<double>::lowest();
    double last_output_VMax_ = 0.0;
};

template <std::size_t D, class SeasOperator> class SeasFaultProbeWriter : public SeasWriter {
public:
    SeasFaultProbeWriter(std::string_view prefix, std::vector<Probe<D>> const& probes,
                         AdaptiveOutputInterval oi, LocalSimplexMesh<D> const& mesh,
                         std::shared_ptr<Curvilinear<D>> cl, std::shared_ptr<SeasOperator> seasop)
        : SeasWriter(prefix, oi), seasop_(std::move(seasop)),
          writer_(prefix, probes, mesh, std::move(cl), seasop_->faultMap(), seasop_->comm()) {}

    void write_step(double time, BlockVector const& state) {
        if (writer_.num_probes() > 0) {
            writer_.write(time, seasop_->state(state, writer_.begin(), writer_.end()));
        }
    }

private:
    std::shared_ptr<SeasOperator> seasop_;
    BoundaryProbeWriter<D> writer_;
};

template <std::size_t D, class SeasOperator> class SeasFaultWriter : public SeasWriter {
public:
    SeasFaultWriter(std::string_view prefix, AdaptiveOutputInterval oi,
                    LocalSimplexMesh<D> const& mesh, std::shared_ptr<Curvilinear<D>> cl,
                    std::shared_ptr<SeasOperator> seasop, unsigned degree)
        : SeasWriter(prefix, oi), seasop_(std::move(seasop)),
          adapter_(mesh, std::move(cl), seasop_->faultMap().localFctNos()), degree_(degree) {}

    void write_step(double time, BlockVector const& state) {
        int rank;
        MPI_Comm_rank(seasop_->comm(), &rank);

        auto writer = VTUWriter<D - 1u>(degree_, true, seasop_->comm());
        writer.addFieldData("time", &time, 1);
        auto piece = writer.addPiece(adapter_);
        piece.addPointData(seasop_->state(state));
        auto base_step = name();
        writer.write(base_step);
        if (rank == 0) {
            pvd_.addTimestep(time, writer.pvtuFileName(base_step));
            pvd_.write(prefix_);
        }
    }

private:
    std::shared_ptr<SeasOperator> seasop_;
    CurvilinearBoundaryVTUAdapter<D> adapter_;
    unsigned degree_;
};

template <std::size_t D, class SeasOperator> class SeasDomainWriter : public SeasWriter {
public:
    SeasDomainWriter(std::string_view prefix, AdaptiveOutputInterval oi,
                     LocalSimplexMesh<D> const& mesh, std::shared_ptr<Curvilinear<D>> cl,
                     std::shared_ptr<SeasOperator> seasop, unsigned degree)
        : SeasWriter(prefix, oi), seasop_(std::move(seasop)),
          adapter_(std::move(cl), seasop_->adapter().numLocalElements()), degree_(degree) {}

    void write_step(double time, BlockVector const& state) {
        int rank;
        MPI_Comm_rank(seasop_->comm(), &rank);

        seasop_->full_solve(time, state);
        auto displacement = seasop_->adapter().displacement();
        auto writer = VTUWriter<D>(degree_, true, seasop_->comm());
        writer.addFieldData("time", &time, 1);
        auto piece = writer.addPiece(adapter_);
        piece.addPointData(displacement);
        auto base_step = name();
        writer.write(base_step);
        if (rank == 0) {
            pvd_.addTimestep(time, writer.pvtuFileName(base_step));
            pvd_.write(prefix_);
        }
    }

private:
    std::shared_ptr<SeasOperator> seasop_;
    CurvilinearVTUAdapter<D> adapter_;
    unsigned degree_;
};

template <class SeasOperator> class SeasMonitor {
public:
    SeasMonitor(std::shared_ptr<SeasOperator> seasop,
                std::vector<std::unique_ptr<SeasWriter>> writers)
        : seasop_(std::move(seasop)), writers_(std::move(writers)) {}

    void monitor(double time, BlockVector const& state) {
        if (!writers_.empty()) {
            double VMax_local = seasop_->VMax_local();
            double VMax;
            MPI_Allreduce(&VMax_local, &VMax, 1, MPI_DOUBLE, MPI_MAX, seasop_->comm());

            for (auto const& writer : writers_) {
                writer->monitor(time, state, VMax);
            }
        }

        if (last_time_ != std::numeric_limits<double>::lowest()) {
            double dt = time - last_time_;
            dt_min_ = std::min(dt_min_, dt);
            dt_max_ = std::max(dt_max_, dt);
        }
        last_time_ = time;
    }

    auto min_time_step() const { return dt_min_; }
    auto max_time_step() const { return dt_max_; }

private:
    std::shared_ptr<SeasOperator> seasop_;
    std::vector<std::unique_ptr<SeasWriter>> writers_;

    double last_time_ = std::numeric_limits<double>::lowest();
    double dt_min_ = std::numeric_limits<double>::max();
    double dt_max_ = std::numeric_limits<double>::lowest();
};

} // namespace tndm

#endif // SEASWRITER_20201006_H
