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

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class SeasAdapterBase {
public:
    SeasAdapterBase(std::shared_ptr<DGOperatorTopo> topo,
                    std::unique_ptr<RefElement<DomainDimension - 1u>> space,
                    std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints,
                    std::array<double, DomainDimension> const& ref_normal);

    std::size_t scratch_mem_size() const {
        return space_->numBasisFunctions() * (2 * DomainDimension * DomainDimension + 1);
    }

    void begin_preparation(std::size_t numFaultFaces, Curvilinear<DomainDimension> const& cl);
    void prepare(std::size_t faultNo, Curvilinear<DomainDimension> const& cl,
                 LinearAllocator<double>& scratch);
    void end_preparation() {}

    auto const& topo() const { return *topo_; }
    auto const& faultMap() const { return faultMap_; }

protected:
    std::shared_ptr<DGOperatorTopo> topo_;
    std::unique_ptr<RefElement<DomainDimension - 1u>> space_;
    BoundaryMap faultMap_;
    std::array<double, DomainDimension> ref_normal_;

    // Basis
    std::vector<Managed<Tensor<double, 3u>>> geoDxi;
    Managed<Matrix<double>> e_q;
    Managed<Matrix<double>> e_q_T;
    Managed<Matrix<double>> minv;

    struct UnitNormal {
        using type = std::array<double, DomainDimension>;
    };
    struct SignFlipped {
        using type = bool;
    };

    using sign_t = mneme::MultiStorage<mneme::DataLayout::SoA, UnitNormal, SignFlipped>;
    mneme::StridedView<sign_t> sign_;
};

} // namespace tndm

#endif // SEASADAPTERBASE_20201102_H
