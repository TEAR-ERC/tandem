#ifndef SEASWRITER_20201006_H
#define SEASWRITER_20201006_H

#include "tandem/AdaptiveOutputStrategy.h"

#include "geometry/Curvilinear.h"
#include "io/PVDWriter.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/LocalSimplexMesh.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace tndm {

template <std::size_t D, class SeasOperator> class SeasWriter {
public:
    SeasWriter(std::string_view baseName, LocalSimplexMesh<D> const& mesh,
               std::shared_ptr<Curvilinear<D>> cl, std::shared_ptr<SeasOperator> seasop,
               unsigned degree, double V_ref, double t_min, double t_max,
               AdaptiveOutputStrategy strategy = AdaptiveOutputStrategy::Threshold)
        : seasop_(std::move(seasop)), fault_adapter_(mesh, cl, seasop_->faultMap().fctNos()),
          adapter_(std::move(cl), seasop_->adapter().numLocalElements()), degree_(degree),
          fault_base_(baseName), base_(baseName), V_ref_(V_ref), t_min_(t_min), t_max_(t_max),
          strategy_(strategy) {
        fault_base_ += "-fault";
        MPI_Comm_rank(seasop_->comm(), &rank_);
    }

    double output_interval(double VMax) const {
        double interval = 0.0;
        switch (strategy_) {
        case AdaptiveOutputStrategy::Threshold:
            interval = VMax >= V_ref_ ? t_min_ : t_max_;
            break;
        case AdaptiveOutputStrategy::Exponential: {
            double falloff = log(t_min_ / t_max_);
            VMax = std::min(V_ref_, VMax);
            interval = t_max_ * exp(falloff * VMax / V_ref_);
            break;
        }
        default:
            break;
        }
        return interval;
    }

    template <class BlockVector> void monitor(double time, BlockVector const& state) {
        double VMax_local = seasop_->VMax_local();
        double VMax;
        MPI_Allreduce(&VMax_local, &VMax, 1, MPI_DOUBLE, MPI_MAX, seasop_->comm());

        auto interval = output_interval(VMax);
        if (time - last_output_time_ >= interval) {
            auto fault_writer = VTUWriter<D - 1u>(degree_, true, seasop_->comm());
            fault_writer.addFieldData("time", &time, 1);
            auto fault_piece = fault_writer.addPiece(fault_adapter_);
            fault_piece.addPointData("state", seasop_->state(state));
            auto fault_base_step = name(fault_base_);
            fault_writer.write(fault_base_step);
            if (rank_ == 0) {
                pvd_fault_.addTimestep(time, fault_writer.pvtuFileName(fault_base_step));
                pvd_fault_.write(fault_base_);
            }

            seasop_->adapter().full_solve(time, state, true);
            auto displacement = seasop_->adapter().displacement();
            auto writer = VTUWriter<D>(degree_, true, seasop_->comm());
            writer.addFieldData("time", &time, 1);
            auto piece = writer.addPiece(adapter_);
            piece.addPointData("u", displacement);
            auto base_step = name(base_);
            writer.write(base_step);
            if (rank_ == 0) {
                pvd_.addTimestep(time, writer.pvtuFileName(base_step));
                pvd_.write(base_);
            }

            ++output_step_;
            last_output_time_ = time;
        }
    }

private:
    std::string name(std::string const& base) const {
        std::stringstream ss;
        ss << base << "_" << output_step_;
        return ss.str();
    }

    std::shared_ptr<SeasOperator> seasop_;
    CurvilinearBoundaryVTUAdapter<D> fault_adapter_;
    CurvilinearVTUAdapter<D> adapter_;
    PVDWriter pvd_;
    PVDWriter pvd_fault_;
    int rank_;

    std::string fault_base_;
    std::string base_;
    unsigned degree_;
    std::size_t output_step_ = 0;
    double last_output_time_ = std::numeric_limits<double>::lowest();

    double V_ref_;
    double t_min_;
    double t_max_;
    AdaptiveOutputStrategy strategy_;
};

} // namespace tndm

#endif // SEASWRITER_20201006_H
