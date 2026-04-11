#ifndef SEASWRITER_20201006_H
#define SEASWRITER_20201006_H

#include "tandem/AdaptiveOutputStrategy.h"

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"
#include "io/BoundaryProbeWriter.h"
#include "io/HDF5Adapter.h"
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
#include <hdf5.h>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tndm::seas {

enum class DataLevel { Scalar, Boundary, Volume, Heirarchichal };

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
    virtual void write(double time, std::vector<double> data) {}

    inline void increase_step(double time, double VMax) {
        ++output_step_;
        last_output_time_ = time;
        last_output_VMax_ = VMax;
    }

    virtual void write_static() {}
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

template <std::size_t D> class MomentRateWriter : public Writer {
public:
    MomentRateWriter(std::string_view prefix, AdaptiveOutputInterval oi,
                     LocalSimplexMesh<D> const& mesh, std::shared_ptr<Curvilinear<D>> cl,
                     unsigned degree, BoundaryMap const& bnd_map, MPI_Comm comm)
        : Writer(prefix, oi), writer_(prefix, comm),
          adapter_(mesh, std::move(cl), bnd_map.localFctNos(), degree), degree_(degree),
          comm_(std::move(comm)) {}

    DataLevel level() const override { return DataLevel::Heirarchichal; }
    bool has_static_writer() const override { return true; }
    void write(double time, std::vector<double> data) override {
        // Get the vertex data from the adapter
        int rank;
        MPI_Comm_rank(comm_, &rank);
        auto numElements = data.size() / (D - 1);

        // Create a dataset for the moment rate
        int extensibleIndex = 1;
        if (momentRateDataset_ == -1) {
            momentRateDataset_ = writer_.createExtendibleDataset(
                "Moment_rate", H5T_IEEE_F64LE, {1, numElements, D - 1},
                {H5S_UNLIMITED, numElements, D - 1}, extensibleIndex);
        }
        // Write the data
        writer_.writeToDataset(momentRateDataset_, H5T_IEEE_F64LE, output_step_, data.data(),
                               {output_step_ + 1, numElements, D - 1}, extensibleIndex);
        // Create a dataset for timestep
        if (timeStepDataset_ == -1) {
            timeStepDataset_ = writer_.createExtendibleDataset("time", H5T_IEEE_F64LE, {1},
                                                               {H5S_UNLIMITED}, extensibleIndex);
        }
        // Write the data
        writer_.writeToDataset(timeStepDataset_, H5T_IEEE_F64LE, output_step_, &time,
                               {output_step_ + 1}, extensibleIndex);
    }
    ~MomentRateWriter() {
        writer_.closeDataset(momentRateDataset_);
        writer_.closeDataset(timeStepDataset_);
    }
    void write_static() override {
        // Get the vertex data from the adapter
        auto faultVertices = adapter_.getVertices();
        auto numFaultBasis = (degree_ + 1) * (degree_ + 2) / 2;
        // Calculate element count

        hsize_t numElements = faultVertices.size() / (numFaultBasis * D);

        // Create a dataset for the vertices
        int extensibleIndex = 0;
        hid_t verticesDataset_ =
            writer_.createExtendibleDataset("faultVertices", H5T_IEEE_F64LE, {numElements, 3, D},
                                            {numElements, 3, D}, extensibleIndex);
        // Write the data
        writer_.writeToDataset(verticesDataset_, H5T_IEEE_F64LE, 0, faultVertices.data(),
                               {numElements, 3, D}, 0);
        // Close dataset when completely done (maybe in destructor)`
        writer_.closeDataset(verticesDataset_);
    }

private:
    HDF5Writer writer_;
    CurvilinearBoundaryHDF5Adapter<D> adapter_;
    unsigned degree_;
    MPI_Comm comm_;
    std::vector<std::array<std::array<double, D>, 3>> faultVertices;
    hid_t momentRateDataset_ = -1;
    hid_t timeStepDataset_ = -1;
};

} // namespace tndm::seas

#endif // SEASWRITER_20201006_H
