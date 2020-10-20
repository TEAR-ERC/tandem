#ifndef SEASWRITER_20201006_H
#define SEASWRITER_20201006_H

#include "geometry/Curvilinear.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/LocalSimplexMesh.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace tndm {

template <std::size_t D, class SeasOperator> class SeasWriter {
public:
    SeasWriter(std::string_view baseName, LocalSimplexMesh<D> const& mesh, Curvilinear<D> const& cl,
               std::shared_ptr<SeasOperator> seasop, unsigned degree)
        : seasop_(std::move(seasop)), fault_adapter_(mesh, cl, seasop_->faultMap().fctNos()),
          adapter_(cl, seasop_->spatialOperator().numLocalElements()), degree_(degree),
          fault_base_(baseName), base_(baseName) {
        fault_base_ += "-fault";
    }

    double output_interval(double VMax) const {
        constexpr double V1 = 0.01;
        constexpr double tmin = 0.1;
        constexpr double tmax = 365 * 24 * 3600;
        double falloff = log(tmin / tmax);
        VMax = std::min(V1, VMax);
        return tmax * exp(falloff * VMax / V1);
    }

    template <class BlockVector> void monitor(double time, BlockVector const& state) {
        auto interval = output_interval(seasop_->VMax());
        if (time - last_output_time_ >= interval) {
            auto fault_writer = VTUWriter<D - 1u>(degree_, true, seasop_->comm());
            fault_writer.addFieldData("time", &time, 1);
            auto fault_piece = fault_writer.addPiece(fault_adapter_);
            fault_piece.addPointData("state", seasop_->state(state));
            fault_writer.write(name(fault_base_));

            auto displacement = seasop_->displacement();
            auto writer = VTUWriter<D>(degree_, true, seasop_->comm());
            writer.addFieldData("time", &time, 1);
            auto piece = writer.addPiece(adapter_);
            piece.addPointData("u", displacement);
            writer.write(name(base_));

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

    std::string fault_base_;
    std::string base_;
    unsigned degree_;
    std::size_t output_step_ = 0;
    double last_output_time_ = std::numeric_limits<double>::lowest();
};

} // namespace tndm

#endif // SEASWRITER_20201006_H
