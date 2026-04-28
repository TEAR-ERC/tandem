#include "Monitor.h"

#include <array>

namespace tndm::seas {

double Monitor::reduce_VMax(double VMax_local, MPI_Comm comm) {
    double VMax;
    MPI_Allreduce(&VMax_local, &VMax, 1, MPI_DOUBLE, MPI_MAX, comm);
    return VMax;
}

void Monitor::update_dt_limits(double time) {
    if (last_time_ != std::numeric_limits<double>::lowest()) {
        double dt = time - last_time_;
        dt_min_ = std::min(dt_min_, dt);
        dt_max_ = std::max(dt_max_, dt);
    }
    last_time_ = time;
}

void MonitorQD::monitor(double time, BlockVector const& state) {
    if (!writers_.empty()) {
        double VMax = reduce_VMax(seasop_->friction().VMax_local(), seasop_->comm());

        bool require_traction = false;
        bool require_displacement = false;
        bool require_stress = false;
        for (auto const& writer : writers_) {
            if (writer->is_write_required(time, VMax)) {
                require_traction = require_traction || writer->level() == DataLevel::Boundary;
                require_displacement = require_displacement || writer->level() == DataLevel::Volume;
                require_stress = require_stress || writer->level() == DataLevel::Stress ||
                                 writer->include_stress();
            }
        }
        // Stress requires displacement to be updated first
        require_displacement = require_displacement || require_stress;
        seasop_->update_internal_state(time, state, !fsal_, require_traction, require_displacement);

        for (auto const& writer : writers_) {
            if (writer->is_write_required(time, VMax)) {
                switch (writer->level()) {
                case DataLevel::Scalar:
                    writer->write(time, mneme::span(&VMax, 1));
                    break;
                case DataLevel::Boundary: {
                    auto data = boundary_data(time, state, writer->subset());
                    writer->write(time, mneme::span(&data, 1));
                    break;
                }
                case DataLevel::Volume: {
                    auto data = volume_data(writer->subset());
                    if (writer->include_stress()) {
                        auto stressDataOpt = stress_data(writer->subset());
                        if (stressDataOpt) {
                            auto combinedData =
                                std::array<FiniteElementFunction<DomainDimension>, 2>{
                                    std::move(data), std::move(*stressDataOpt)};
                            writer->write(time,
                                          mneme::span(combinedData.data(), combinedData.size()));
                            break;
                        }
                    }
                    writer->write(time, mneme::span(&data, 1));
                    break;
                }
                case DataLevel::Stress: {
                    auto data_opt = stress_data(writer->subset());
                    if (data_opt) {
                        auto& data = *data_opt;
                        writer->write(time, mneme::span(&data, 1));
                    } else {
                        // Do nothing
                        // TODO: Decide on a warning
                    }
                    break;
                }
                };
                writer->increase_step(time, VMax);
            }
        }
    }

    update_dt_limits(time);
}

void MonitorQD::write_static() {
    for (auto const& writer : writers_) {
        switch (writer->level()) {
        case DataLevel::Scalar:
            break;
        case DataLevel::Boundary: {
            auto data = static_boundary_data(writer->subset());
            writer->write_static(mneme::span(&data, 1));
            break;
        }
        case DataLevel::Volume: {
            auto data = static_volume_data(writer->subset());
            writer->write_static(mneme::span(&data, 1));
            break;
        }
        case DataLevel::Stress:
            // Do nothing
            // TODO: Decide on a warning
            break;
        };
    }
}

void MonitorFD::monitor(double time, BlockVector const& v, BlockVector const& u,
                        BlockVector const& s) {
    if (!writers_.empty()) {
        double VMax = reduce_VMax(seasop_->friction().VMax_local(), seasop_->comm());

        for (auto const& writer : writers_) {
            if (writer->is_write_required(time, VMax)) {
                switch (writer->level()) {
                case DataLevel::Scalar:
                    writer->write(time, mneme::span(&VMax, 1));
                    break;
                case DataLevel::Boundary: {
                    auto data = boundary_data(time, s, writer->subset());
                    writer->write(time, mneme::span(&data, 1));
                    break;
                }
                case DataLevel::Volume: {
                    auto data = volume_data(v, u, writer->subset());
                    data[0].setNames(velocity_names(data[0].numQuantities()));
                    writer->write(time, mneme::span(data.data(), 2));
                    break;
                }
                case DataLevel::Stress:
                    // Stress output not supported for fully dynamic simulations
                    // TODO: Decide on a warning
                    break;
                };
                writer->increase_step(time, VMax);
            }
        }
    }

    update_dt_limits(time);
}

void MonitorFD::write_static() {
    for (auto const& writer : writers_) {
        switch (writer->level()) {
        case DataLevel::Scalar:
            break;
        case DataLevel::Boundary: {
            auto data = static_boundary_data(writer->subset());
            writer->write_static(mneme::span(&data, 1));
            break;
        }
        case DataLevel::Volume: {
            auto data = static_volume_data(writer->subset());
            writer->write_static(mneme::span(&data, 1));
            break;
        }
        case DataLevel::Stress:
            // No static stress output
            // TODO: Decide on a warning
            break;
        };
    }
}

auto MonitorFD::velocity_names(std::size_t numQuantities) -> std::vector<std::string> {
    auto names = std::vector<std::string>(numQuantities);
    char buf[100];
    for (std::size_t q = 0; q < numQuantities; ++q) {
        snprintf(buf, sizeof(buf), "v%lu", q);
        names[q] = buf;
    }
    return names;
}

} // namespace tndm::seas
