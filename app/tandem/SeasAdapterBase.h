#ifndef SEASADAPTERBASE_20201102_H
#define SEASADAPTERBASE_20201102_H

#include "config.h"

#include "form/BoundaryMap.h"
#include "form/DGOperatorTopo.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class SeasAdapterBase {
public:
    SeasAdapterBase(std::unique_ptr<BoundaryMap> fault_map,
                    std::shared_ptr<Curvilinear<DomainDimension>> cl,
                    std::shared_ptr<DGOperatorTopo> topo,
                    std::unique_ptr<RefElement<DomainDimension - 1u>> space,
                    std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints,
                    std::array<double, DomainDimension> const& up,
                    std::array<double, DomainDimension> const& ref_normal);

    std::size_t scratch_mem_size() const {
        return nq_ * (2 * DomainDimension * DomainDimension + DomainDimension + 1);
    }

    void begin_preparation(std::size_t numFaultFaces);
    void prepare(std::size_t faultNo, LinearAllocator<double>& scratch);
    void end_preparation() {}

    auto const& topo() const { return *topo_; }
    auto const& faultMap() const { return *faultMap_; }

protected:
    std::unique_ptr<BoundaryMap> faultMap_;
    std::shared_ptr<Curvilinear<DomainDimension>> cl_;
    std::shared_ptr<DGOperatorTopo> topo_;
    std::unique_ptr<RefElement<DomainDimension - 1u>> space_;
    std::array<double, DomainDimension> up_;
    std::array<double, DomainDimension> ref_normal_;
    std::size_t nq_;

    // Basis
    std::vector<Managed<Tensor<double, 3u>>> geoDxi_q;
    Managed<Matrix<double>> e_q;
    Managed<Matrix<double>> e_q_T;
    Managed<Matrix<double>> minv;

    struct SignFlipped {
        using type = bool;
    };
    struct UnitNormal {
        using type = std::array<double, DomainDimension>;
    };
    struct FaultBasis {
        using type = std::array<double, DomainDimension * DomainDimension>;
    };

    using fault_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, SignFlipped, UnitNormal, FaultBasis>;
    mneme::StridedView<fault_t> fault_;
};

} // namespace tndm

#endif // SEASADAPTERBASE_20201102_H
