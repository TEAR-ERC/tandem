#ifndef SEASMONITOR_20210909_H
#define SEASMONITOR_20210909_H

#include "config.h"
#include "form/FiniteElementFunction.h"
#include "form/SeasFDOperator.h"
#include "form/SeasQDOperator.h"
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
    double reduce_VMax(double VMax_local, MPI_Comm comm);

    void update_dt_limits(double time);

    std::vector<std::unique_ptr<Writer>> writers_;
    bool fsal_;

private:
    double last_time_ = std::numeric_limits<double>::lowest();
    double dt_min_ = std::numeric_limits<double>::max();
    double dt_max_ = std::numeric_limits<double>::lowest();
};

class MonitorQD : public Monitor {
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
    void monitor(double time, BlockVector const& state);

private:
    inline auto boundary_data(BlockVector const& state, std::vector<std::size_t> const* subset)
        -> FiniteElementFunction<DomainDimension - 1u> {
        if (subset) {
            return seasop_->state(state, *subset);
        } else {
            return seasop_->state(state);
        }
    }
    inline auto volume_data(std::vector<std::size_t> const* subset)
        -> FiniteElementFunction<DomainDimension> {
        if (subset) {
            return seasop_->displacement(*subset);
        } else {
            return seasop_->displacement();
        }
    }

    std::shared_ptr<SeasQDOperator> seasop_;
};

class MonitorFD : public Monitor {
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
    void monitor(double time, BlockVector const& v, BlockVector const& u, BlockVector const& s);

private:
    inline auto boundary_data(BlockVector const& s, std::vector<std::size_t> const* subset)
        -> FiniteElementFunction<DomainDimension - 1u> {
        if (subset) {
            return seasop_->fault_state(s, *subset);
        } else {
            return seasop_->fault_state(s);
        }
    }
    inline auto volume_data(BlockVector const& v, BlockVector const& u,
                            std::vector<std::size_t> const* subset)
        -> std::array<FiniteElementFunction<DomainDimension>, 2> {
        if (subset) {
            return {seasop_->domain_function(v, *subset), seasop_->domain_function(u, *subset)};
        } else {
            return {seasop_->domain_function(v), seasop_->domain_function(u)};
        }
    }
    auto velocity_names(std::size_t numQuantities) -> std::vector<std::string>;

    std::shared_ptr<SeasFDOperator> seasop_;
};

} // namespace tndm::seas

#endif // SEASMONITOR_20210909_H
