#ifndef ADAPTERBASE_20210906_H
#define ADAPTERBASE_20210906_H

#include "config.h"

#include "form/BoundaryMap.h"
#include "form/DGOperatorTopo.h"
#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "quadrules/SimplexQuadratureRule.h"
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

class AdapterBase {
public:
    AdapterBase(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                std::unique_ptr<RefElement<DomainDimension - 1u>> space,
                SimplexQuadratureRule<DomainDimension - 1u> const& quad_rule,
                std::array<double, DomainDimension> const& up,
                std::array<double, DomainDimension> const& ref_normal);

    std::size_t scratch_mem_size() const {
        return quad_rule_.size() * (2 * DomainDimension * DomainDimension + DomainDimension + 1);
    }
    auto ref_normal() const { return ref_normal_; }

    void begin_preparation(std::size_t numFaultFaces);
    void prepare(std::size_t faultNo, FacetInfo const& info, LinearAllocator<double>& scratch);
    void end_preparation() {}

protected:
    std::shared_ptr<Curvilinear<DomainDimension>> cl_;
    std::unique_ptr<RefElement<DomainDimension - 1u>> space_;
    SimplexQuadratureRule<DomainDimension - 1u> quad_rule_;
    std::array<double, DomainDimension> up_;
    std::array<double, DomainDimension> ref_normal_;

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

#endif // ADAPTERBASE_20210906_H
