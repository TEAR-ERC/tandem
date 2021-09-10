#ifndef SEASMONITOR_20210909_H
#define SEASMONITOR_20210909_H

#include "config.h"
#include "form/FiniteElementFunction.h"
#include "tandem/Writer.h"

#include "interface/BlockVector.h"

#include <mpi.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace tndm::seas {

class Monitor {
public:
    /**
     * @brief Construct a basic Monitor
     *
     * @param fsal Flag indicating whether time integrator has First Same As Last property
     */
    Monitor(bool fsal) : fsal_(fsal) {}

    void add_writer(std::unique_ptr<Writer> writer) { writers_.emplace_back(std::move(writer)); }

    auto min_time_step() const { return dt_min_; }
    auto max_time_step() const { return dt_max_; }

protected:
    double reduce_VMax(double VMax_local, MPI_Comm comm) {
        double VMax;
        MPI_Allreduce(&VMax_local, &VMax, 1, MPI_DOUBLE, MPI_MAX, comm);
        return VMax;
    }

    void update_dt_limits(double time) {
        if (last_time_ != std::numeric_limits<double>::lowest()) {
            double dt = time - last_time_;
            dt_min_ = std::min(dt_min_, dt);
            dt_max_ = std::max(dt_max_, dt);
        }
        last_time_ = time;
    }

    std::vector<std::unique_ptr<Writer>> writers_;
    bool fsal_;

private:
    double last_time_ = std::numeric_limits<double>::lowest();
    double dt_min_ = std::numeric_limits<double>::max();
    double dt_max_ = std::numeric_limits<double>::lowest();
};

template <class SeasQDOperator> class MonitorQD : public Monitor {
public:
    /**
     * @brief Construct a Monitor
     *
     * @param seasop Seas operator
     * @param fsal Flag indicating whether time integrator has First Same As Last property
     */
    MonitorQD(std::shared_ptr<SeasQDOperator> seasop, bool fsal)
        : Monitor(fsal), seasop_(std::move(seasop)) {}

    /**
     * @brief Monitor function called by time integrator
     *
     * @param time Current time
     * @param state Quantities of time integrator
     */
    void monitor(double time, BlockVector const& state) {
        if (!writers_.empty()) {
            double VMax = reduce_VMax(seasop_->friction().VMax_local(), seasop_->comm());

            bool require_traction = false;
            bool require_displacement = false;
            for (auto const& writer : writers_) {
                if (writer->is_write_required(time, VMax)) {
                    require_traction = require_traction || writer->level() == DataLevel::Boundary;
                    require_displacement =
                        require_displacement || writer->level() == DataLevel::Volume;
                }
            }
            seasop_->update_internal_state(time, state, !fsal_, require_traction,
                                           require_displacement);

            for (auto const& writer : writers_) {
                if (writer->is_write_required(time, VMax)) {
                    switch (writer->level()) {
                    case DataLevel::Scalar:
                        writer->write(time, mneme::span(&VMax, 1));
                        break;
                    case DataLevel::Boundary: {
                        auto data = boundary_data(state, writer->subset());
                        writer->write(time, mneme::span(&data, 1));
                        break;
                    }
                    case DataLevel::Volume: {
                        auto data = volume_data(writer->subset());
                        writer->write(time, mneme::span(&data, 1));
                        break;
                    }
                    };
                    writer->increase_step(time, VMax);
                }
            }
        }

        update_dt_limits(time);
    }

private:
    auto boundary_data(BlockVector const& state, std::vector<std::size_t> const* subset) {
        if (subset) {
            return seasop_->state(state, *subset);
        } else {
            return seasop_->state(state);
        }
    }
    auto volume_data(std::vector<std::size_t> const* subset) {
        if (subset) {
            return seasop_->displacement(*subset);
        } else {
            return seasop_->displacement();
        }
    }

    std::shared_ptr<SeasQDOperator> seasop_;
};

template <class SeasFDOperator> class MonitorFD : public Monitor {
public:
    /**
     * @brief Construct a Monitor
     *
     * @param seasop Seas operator
     * @param fsal Flag indicating whether time integrator has First Same As Last property
     */
    MonitorFD(std::shared_ptr<SeasFDOperator> seasop, bool fsal)
        : Monitor(fsal), seasop_(std::move(seasop)) {}

    /**
     * @brief Monitor function called by time integrator
     *
     * @param time Current time
     * @param v Velocity vector
     * @param u Displacement vector
     * @param s Slip and state vector
     */
    void monitor(double time, BlockVector const& v, BlockVector const& u, BlockVector const& s) {
        if (!writers_.empty()) {
            double VMax = reduce_VMax(seasop_->friction().VMax_local(), seasop_->comm());

            for (auto const& writer : writers_) {
                if (writer->is_write_required(time, VMax)) {
                    switch (writer->level()) {
                    case DataLevel::Scalar:
                        writer->write(time, mneme::span(&VMax, 1));
                        break;
                    case DataLevel::Boundary: {
                        auto data = boundary_data(s, writer->subset());
                        writer->write(time, mneme::span(&data, 1));
                        break;
                    }
                    case DataLevel::Volume: {
                        auto data = volume_data(v, u, writer->subset());
                        data[0].setNames(velocity_names(data[0].numQuantities()));
                        writer->write(time, mneme::span(data.data(), 2));
                        break;
                    }
                    };
                    writer->increase_step(time, VMax);
                }
            }
        }

        update_dt_limits(time);
    }

private:
    auto boundary_data(BlockVector const& s, std::vector<std::size_t> const* subset) {
        if (subset) {
            return seasop_->fault_state(s, *subset);
        } else {
            return seasop_->fault_state(s);
        }
    }
    auto volume_data(BlockVector const& v, BlockVector const& u,
                     std::vector<std::size_t> const* subset)
        -> std::array<FiniteElementFunction<DomainDimension>, 2> {
        if (subset) {
            return {seasop_->domain_function(v, *subset), seasop_->domain_function(u, *subset)};
        } else {
            return {seasop_->domain_function(v), seasop_->domain_function(u)};
        }
    }
    auto velocity_names(std::size_t numQuantities) {
        auto names = std::vector<std::string>(numQuantities);
        char buf[100];
        for (std::size_t q = 0; q < numQuantities; ++q) {
            snprintf(buf, sizeof(buf), "v%lu", q);
            names[q] = buf;
        }
        return names;
    }

    std::shared_ptr<SeasFDOperator> seasop_;
};

} // namespace tndm::seas

#endif // SEASMONITOR_20210909_H
