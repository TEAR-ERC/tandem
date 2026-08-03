#ifndef SEASWRITER_20201006_H
#define SEASWRITER_20201006_H

#include "tandem/AdaptiveOutputStrategy.h"

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"
#include "io/BoundaryProbeWriter.h"
#include "io/HDF5Adapter.h"
#include "io/HDF5ProbeWriter.h"
#include "io/HDF5Writer.h"
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
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tndm::seas {

enum class DataLevel {
    Scalar,
    Boundary,
    Volume,
    BoundaryForMomentRate // Fault surface data (moment rate) computed over fault area
};

class Writer {
public:
    Writer(std::string_view prefix, AdaptiveOutputInterval oi) : prefix_(prefix), oi_(oi) {}
    virtual ~Writer() {}

    virtual DataLevel level() const = 0;
    virtual std::vector<std::size_t> const* subset() const { return nullptr; }
    virtual bool has_static_writer() const { return false; }

    /// True when this writer produces one event_N folder per v_th excursion (see name()).
    inline bool event_mode() const { return oi_.v_th().has_value() && oi_.high_freq(); }

    inline bool is_write_required(double time, double VMax) const {
        if (auto v_th = oi_.v_th()) {
            if (oi_.high_freq()) {
                // High-frequency mode: write on every monitor step for as long as
                // VMax stays at or above the threshold, resuming the single
                // edge-triggered behavior once VMax drops back below it.
                return VMax >= *v_th;
            }
            // Threshold triggered output: write once each time VMax rises through the
            // threshold. last_seen_VMax_ holds VMax from the previous monitor
            // step, so this fires only on the upward crossing and resets once
            // VMax has dropped back below the threshold. All other (adaptive)
            // output is suppressed in this mode.
            return last_seen_VMax_ < *v_th && VMax >= *v_th;
        }
        double delta_time = time - last_output_time_;
        return oi_(delta_time, last_output_VMax_, VMax);
    }

    /**
     * @brief Record VMax for the current monitor step.
     *
     * Must be called exactly once per monitor step, after all
     * is_write_required() queries for that step, so that the next step can
     * detect a rising threshold crossing.
     */
    inline void observe(double VMax) { last_seen_VMax_ = VMax; }

    /**
     * @brief Advance event-folder bookkeeping for high_freq + v_th output.
     *
     * Must be called exactly once per monitor step, for every writer, before
     * any is_write_required()/write() calls for that step (i.e. before
     * observe() updates last_seen_VMax_). Detects the rising edge of a new
     * v_th excursion so that name() can immediately target a fresh event_N
     * folder starting at step 0. No-op unless both v_th and high_freq are set.
     */
    inline void prepare_step(double VMax) {
        if (!event_mode()) {
            return;
        }
        double v_th = *oi_.v_th();
        bool is_new_event = last_seen_VMax_ < v_th && VMax >= v_th;
        if (is_new_event) {
            if (event_started_) {
                ++event_index_;
            }
            event_started_ = true;
            event_step_ = 0;
        }
    }

    virtual void write(double time, mneme::span<double> data) {}
    virtual void write(double time, mneme::span<FiniteElementFunction<1u>> data) {}
    virtual void write(double time, mneme::span<FiniteElementFunction<2u>> data) {}
    virtual void write(double time, mneme::span<FiniteElementFunction<3u>> data) {}
    virtual void write(double time, std::vector<double> const& data) {}

    virtual void increase_step(double time, double VMax) {
        ++output_step_;
        ++event_step_;
        last_output_time_ = time;
        last_output_VMax_ = VMax;
    }

    virtual void write_static() {}
    virtual void write_static(mneme::span<double> data) {}
    virtual void write_static(mneme::span<FiniteElementFunction<1u>> data) {}
    virtual void write_static(mneme::span<FiniteElementFunction<2u>> data) {}
    virtual void write_static(mneme::span<FiniteElementFunction<3u>> data) {}

protected:
    // Every excursion of VMax above v_th (until it drops back below) is one "event". In
    // high_freq mode, each event's files are grouped into their own event_N subfolder
    // (event_0, event_1, ...), numbered in the order events occur, with step numbering
    // (the "_<step>" filename suffix) restarting at 0 for each event.
    inline std::string name() const {
        std::stringstream ss;
        if (event_mode()) {
            std::filesystem::path prefix_path(prefix_);
            auto parent =
                prefix_path.has_parent_path() ? prefix_path.parent_path() : std::filesystem::current_path();
            auto dir = parent / ("event_" + std::to_string(event_index_));
            std::filesystem::create_directories(dir);
            ss << (dir / prefix_path.filename()).string() << "_" << event_step_;
        } else {
            ss << prefix_ << "_" << output_step_;
        }
        return ss.str();
    }

    std::string prefix_;
    AdaptiveOutputInterval oi_;

    std::size_t output_step_ = 0;
    double last_output_time_ = std::numeric_limits<double>::lowest();
    double last_output_VMax_ = 0.0;
    double last_seen_VMax_ = 0.0;

    std::size_t event_index_ = 0;
    std::size_t event_step_ = 0;
    bool event_started_ = false;
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

private:
    PVDWriter pvd_;
    CurvilinearVTUAdapter<D> adapter_;
    unsigned degree_;
    MPI_Comm comm_;
    bool jacobian_;
};

#ifdef ENABLE_HDF5
template <std::size_t D> class MomentRateWriter : public Writer {
public:
    MomentRateWriter(std::string_view prefix, AdaptiveOutputInterval oi,
                     LocalSimplexMesh<D> const& mesh, std::shared_ptr<Curvilinear<D>> cl,
                     unsigned degree, BoundaryMap const& bnd_map, MPI_Comm comm,
                     bool checkpoint_enabled)
        : Writer(prefix, oi), writer_(prefix, comm, checkpoint_enabled),
          adapter_(mesh, std::move(cl), bnd_map.localFctNos(), degree) {
        // Check if time dataset exists from previous run (checkpoint restart)
        htri_t dataset_exists = H5Lexists(writer_.file(), "time", H5P_DEFAULT);
        if (dataset_exists > 0) {
            // Open existing time dataset and get its extent
            hid_t time_dset = H5Dopen(writer_.file(), "time", H5P_DEFAULT);
            if (time_dset >= 0) {
                hid_t space = H5Dget_space(time_dset);
                hsize_t dims[1];
                H5Sget_simple_extent_dims(space, dims, nullptr);
                output_step_ = dims[0]; // Resume from where we left off
                H5Sclose(space);
                H5Dclose(time_dset);
            }
        }
    }

    DataLevel level() const override { return DataLevel::BoundaryForMomentRate; }
    bool has_static_writer() const override { return true; }
    void write(double time, std::vector<double> const& data) override {
        // Get the vertex data from the adapter
        auto numElements = data.size() / (D - 1);

        // Create a dataset for the moment rate
        int glueDimensionMoment = 0;
        int extensibleDimensionMoment = 1;

        if (momentRateDataset_ == -1) {
            momentRateDataset_ = writer_.createExtendibleDataset(
                "momentRate", H5T_IEEE_F64LE, {numElements, 1, D - 1},
                {numElements, H5S_UNLIMITED, D - 1}, glueDimensionMoment);
        }
        // Write the data
        writer_.writeToDataset(momentRateDataset_, H5T_IEEE_F64LE, output_step_, data.data(),
                               {numElements, output_step_ + 1, D - 1}, glueDimensionMoment,
                               extensibleDimensionMoment);
        // Create a dataset for timestep
        int glueDimensionTimeStep = 0;
        int extensibleDimensionTimeStep = 0;
        bool isDistributed = false;
        if (timeStepDataset_ == -1) {
            timeStepDataset_ = writer_.createExtendibleDataset(
                "time", H5T_IEEE_F64LE, {1}, {H5S_UNLIMITED}, glueDimensionTimeStep, isDistributed);
        }
        // Write the data
        writer_.writeToDataset(timeStepDataset_, H5T_IEEE_F64LE, output_step_, &time,
                               {output_step_ + 1}, glueDimensionTimeStep,
                               extensibleDimensionTimeStep, isDistributed);
    }
    ~MomentRateWriter() {
        if (momentRateDataset_ != -1) {
            writer_.closeDataset(momentRateDataset_);
        }
        if (timeStepDataset_ != -1) {
            writer_.closeDataset(timeStepDataset_);
        }
    }
    void write_static() override {
        // Get the vertex data from the adapter
        auto faultVertices = adapter_.getVertices();
        auto numFaultBasis = adapter_.getNumBasisNodes();
        // Calculate element count

        hsize_t numElements = faultVertices.size() / (numFaultBasis * D);

        // Create a dataset for the vertices
        int glueDimension = 0;
        int extensibleDimension = 0;
        hid_t vertices_dset =
            writer_.createExtendibleDataset("faultVertices", H5T_IEEE_F64LE, {numElements, D, D},
                                            {numElements, D, D}, glueDimension);
        // Write the data
        writer_.writeToDataset(vertices_dset, H5T_IEEE_F64LE, 0, faultVertices.data(),
                               {numElements, D, D}, glueDimension, extensibleDimension);
        writer_.closeDataset(vertices_dset);

        auto globalFctNos = adapter_.getGlobalFctNos();
        hsize_t numFcts = globalFctNos.size();
        hid_t fct_no_dset = writer_.createExtendibleDataset("faultNo", H5T_NATIVE_INT, {numFcts},
                                                            {numFcts}, extensibleDimension);

        writer_.writeToDataset(fct_no_dset, H5T_NATIVE_LLONG, 0, globalFctNos.data(), {numFcts},
                               glueDimension, extensibleDimension);

        writer_.closeDataset(fct_no_dset);
    }
    void increase_step(double time, double VMax) override {
        if (!received_first_step_) {
            received_first_step_ = true;
            return;
        }
        Writer::increase_step(time, VMax);
    }

private:
    HDF5Writer writer_;
    CurvilinearBoundaryHDF5Adapter<D> adapter_;
    hid_t momentRateDataset_ = -1;
    hid_t timeStepDataset_ = -1;
    bool received_first_step_ = false;
};
#endif

template <std::size_t D, bool isBoundary> class HDF5CommonProbeWriter : public Writer {
public:
    HDF5CommonProbeWriter(std::string_view prefix, std::vector<Probe<D>> const& probes,
                          AdaptiveOutputInterval oi, LocalSimplexMesh<D> const& mesh,
                          std::shared_ptr<Curvilinear<D>> cl, BoundaryMap const& bnd_map,
                          MPI_Comm comm, bool checkpoint_enabled)
        : Writer(prefix, oi),
          writer_(prefix, probes, mesh, std::move(cl), bnd_map, comm, checkpoint_enabled) {}

    DataLevel level() const override {
        if constexpr (isBoundary) {
            return DataLevel::Boundary;
        } else {
            return DataLevel::Volume;
        }
    }
    void write_static(mneme::span<FiniteElementFunction<isBoundary ? D - 1u : D>> data) override {
        writer_.initialize_datasets(data);
    }
    std::vector<std::size_t> const* subset() const override { return &writer_.bndNos(); }
    void write(double time,
               mneme::span<FiniteElementFunction<isBoundary ? D - 1u : D>> data) override {
        writer_.write(time, std::move(data), output_step_);
    }

private:
    HDF5ProbeWriter<D, isBoundary> writer_;
};

} // namespace tndm::seas

#endif // SEASWRITER_20201006_H
