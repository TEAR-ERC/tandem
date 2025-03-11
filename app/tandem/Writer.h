#ifndef SEASWRITER_20201006_H
#define SEASWRITER_20201006_H

#include "tandem/AdaptiveOutputStrategy.h"

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"
#include "io/BoundaryProbeWriter.h"
#include "io/PVDWriter.h"
#include "io/ProbeWriter.h"
#include "io/ScalarWriter.h"
#include "io/TableWriter.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/LocalSimplexMesh.h"

#include <mneme/span.hpp>
#include <mpi.h>

#include <cstddef>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace tndm::seas {

enum class DataLevel { Scalar, Boundary, Volume };

class Writer {
public:
    Writer(std::string_view prefix, AdaptiveOutputInterval oi) : prefix_(prefix), oi_(oi) {}
    virtual ~Writer() {}

    virtual DataLevel level() const = 0;
    virtual std::vector<std::size_t> const* subset() const { return nullptr; }
    virtual bool has_static_writer() const { return false; }

    inline bool is_write_required(double time, double VMax) const {
        double delta_time = time - last_output_time_;
        return oi_(delta_time, last_output_VMax_, VMax);
    }

    virtual void write(double time, mneme::span<double> data) {}
    virtual void write(double time, mneme::span<FiniteElementFunction<1u>> data) {}
    virtual void write(double time, mneme::span<FiniteElementFunction<2u>> data) {}
    virtual void write(double time, mneme::span<FiniteElementFunction<3u>> data) {}

    inline void increase_step(double time, double VMax) {

        ++output_step_;
        last_output_time_ = time;
        last_output_VMax_ = VMax;

        // Open a file for appending (creates the file if it doesn't exist)
        std::ofstream output_file(prefix_ + "_status.txt");
        if (output_file.is_open()) {
            output_file << output_step_ << " " << std::scientific
                        << std::setprecision(std::numeric_limits<double>::digits10 + 1) << time
                        << " " << VMax << std::endl;
        } else {
            std::cerr << "Error opening file for writing!" << std::endl;
        }
        output_file.close();
    }

    virtual void set_state(std::size_t step, double time, double VMax) {
        output_step_ = step;
        last_output_time_ = time;
        last_output_VMax_ = VMax;
    }

    virtual void write_static(mneme::span<double> data) {}
    virtual void write_static(mneme::span<FiniteElementFunction<1u>> data) {}
    virtual void write_static(mneme::span<FiniteElementFunction<2u>> data) {}
    virtual void write_static(mneme::span<FiniteElementFunction<3u>> data) {}

protected:
    inline std::string name() const {
        std::stringstream ss;
        ss << prefix_ << "_" << output_step_;
        return ss.str();
    }

    std::string prefix_;
    AdaptiveOutputInterval oi_;

    std::size_t output_step_ = 0;
    double last_output_time_ = std::numeric_limits<double>::lowest();
    double last_output_VMax_ = 0.0;
};

template <std::size_t D> class FaultProbeWriter : public Writer {
public:
    FaultProbeWriter(std::string_view prefix, std::unique_ptr<TableWriter> table_writer,
                     std::vector<Probe<D>> const& probes, AdaptiveOutputInterval oi,
                     LocalSimplexMesh<D> const& mesh, std::shared_ptr<Curvilinear<D>> cl,
                     BoundaryMap const& bnd_map, MPI_Comm comm)
        : Writer(prefix, oi),
          writer_(prefix, std::move(table_writer), probes, mesh, std::move(cl), bnd_map, comm) {}

    DataLevel level() const override { return DataLevel::Boundary; }
    std::vector<std::size_t> const* subset() const override { return &writer_.bndNos(); }
    void write(double time, mneme::span<FiniteElementFunction<D - 1u>> data) override {
        if (writer_.num_probes() > 0) {
            writer_.write(time, std::move(data));
        }
    }
    inline void set_state(std::size_t step, double time, double VMax) override {
        Writer::set_state(step, time, VMax);
        writer_.truncate_after_restart(step);
    }

private:
    BoundaryProbeWriter<D> writer_;
};

template <std::size_t D> class DomainProbeWriter : public Writer {
public:
    DomainProbeWriter(std::string_view prefix, std::unique_ptr<TableWriter> table_writer,
                      std::vector<Probe<D>> const& probes, AdaptiveOutputInterval oi,
                      LocalSimplexMesh<D> const& mesh, std::shared_ptr<Curvilinear<D>> cl,
                      MPI_Comm comm)
        : Writer(prefix, oi),
          writer_(prefix, std::move(table_writer), probes, mesh, std::move(cl), comm) {}

    DataLevel level() const override { return DataLevel::Volume; }
    std::vector<std::size_t> const* subset() const override { return &writer_.elNos(); }
    void write(double time, mneme::span<FiniteElementFunction<D>> data) override {
        if (writer_.num_probes() > 0) {
            writer_.write(time, std::move(data));
        }
    }
    inline void set_state(std::size_t step, double time, double VMax) override {
        Writer::set_state(step, time, VMax);
        writer_.truncate_after_restart(step);
    }

private:
    ProbeWriter<D> writer_;
};

template <std::size_t D> class FaultWriter : public Writer {
public:
    FaultWriter(std::string_view prefix, AdaptiveOutputInterval oi, LocalSimplexMesh<D> const& mesh,
                std::shared_ptr<Curvilinear<D>> cl, unsigned degree, BoundaryMap const& bnd_map,
                MPI_Comm comm)
        : Writer(prefix, oi), pvd_(prefix), adapter_(mesh, std::move(cl), bnd_map.localFctNos()),
          degree_(degree), comm_(std::move(comm)) {}

    DataLevel level() const override { return DataLevel::Boundary; }
    bool has_static_writer() const override { return true; }
    void write(double time, mneme::span<FiniteElementFunction<D - 1u>> data) override {
        int rank;
        MPI_Comm_rank(comm_, &rank);

        auto writer = VTUWriter<D - 1u>(degree_, true, comm_);
        writer.addFieldData("time", &time, 1);
        auto& piece = writer.addPiece(adapter_);
        for (auto const& fun : data) {
            piece.addPointData(fun);
        }
        auto base_step = this->name();
        writer.write(base_step);
        if (rank == 0) {
            pvd_.addTimestep(time, writer.pvtuFileName(base_step));
            pvd_.write();
        }
    }

    void write_static(mneme::span<FiniteElementFunction<D - 1u>> data) override {
        auto writer = VTUWriter<D - 1u>(degree_, true, comm_);
        auto& piece = writer.addPiece(adapter_);
        for (auto const& fun : data) {
            piece.addPointData(fun);
        }
        writer.write(prefix_ + "-static");
    }

    inline void set_state(std::size_t step, double time, double VMax) override {
        Writer::set_state(step, time, VMax);
        int rank;
        MPI_Comm_rank(comm_, &rank);
        if (rank == 0) {
            pvd_.truncate_after_restart(step);
        }
    }

private:
    PVDWriter pvd_;
    CurvilinearBoundaryVTUAdapter<D> adapter_;
    unsigned degree_;
    MPI_Comm comm_;
};

class FaultScalarWriter : public Writer {
public:
    FaultScalarWriter(std::string_view prefix, std::unique_ptr<TableWriter> table_writer,
                      AdaptiveOutputInterval oi, MPI_Comm comm)
        : Writer(prefix, oi), writer_(prefix, std::move(table_writer), {"VMax"}),
          comm_(std::move(comm)) {}

    DataLevel level() const override { return DataLevel::Scalar; }
    void write(double time, mneme::span<double> data) override {
        int rank;
        MPI_Comm_rank(comm_, &rank);

        if (rank == 0) {
            writer_.write(time, std::move(data));
        }
    }

private:
    ScalarWriter writer_;
    MPI_Comm comm_;
};

template <std::size_t D> class DomainWriter : public Writer {
public:
    DomainWriter(std::string_view prefix, AdaptiveOutputInterval oi,
                 LocalSimplexMesh<D> const& mesh, std::shared_ptr<Curvilinear<D>> cl,
                 unsigned degree, bool jacobian, MPI_Comm comm)
        : Writer(prefix, oi), pvd_(prefix), adapter_(std::move(cl), mesh.elements().localSize()),
          degree_(degree), jacobian_(jacobian), comm_(std::move(comm)) {}

    DataLevel level() const override { return DataLevel::Volume; }
    void write(double time, mneme::span<FiniteElementFunction<D>> data) override {
        int rank;
        MPI_Comm_rank(comm_, &rank);

        auto writer = VTUWriter<D>(degree_, true, comm_);
        writer.addFieldData("time", &time, 1);
        auto& piece = writer.addPiece(adapter_);
        for (auto const& fun : data) {
            piece.addPointData(fun);
            if (jacobian_) {
                piece.addJacobianData(fun, adapter_);
            }
        }
        auto base_step = this->name();
        writer.write(base_step);
        if (rank == 0) {
            pvd_.addTimestep(time, writer.pvtuFileName(base_step));
            pvd_.write();
        }
    }

    void write_static(mneme::span<FiniteElementFunction<D>> data) override {
        auto writer = VTUWriter<D>(degree_, true, comm_);
        auto& piece = writer.addPiece(adapter_);
        for (auto const& fun : data) {
            piece.addPointData(fun);
        }
        writer.write(prefix_ + "-static");
    }

    inline void set_state(std::size_t step, double time, double VMax) override {
        Writer::set_state(step, time, VMax);
        int rank;
        MPI_Comm_rank(comm_, &rank);
        if (rank == 0) {
            pvd_.truncate_after_restart(step);
        }
    }

private:
    PVDWriter pvd_;
    CurvilinearVTUAdapter<D> adapter_;
    unsigned degree_;
    MPI_Comm comm_;
    bool jacobian_;
};

} // namespace tndm::seas

#endif // SEASWRITER_20201006_H
