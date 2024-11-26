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
                RefElement<DomainDimension - 1u> const& space,
                SimplexQuadratureRule<DomainDimension - 1u> const& quad_rule,
                std::array<double, DomainDimension> const& up,
                std::array<double, DomainDimension> const& ref_normal);

    inline std::size_t scratch_mem_size() const {
        return quad_rule_.size() * (2 * DomainDimension * DomainDimension + DomainDimension + 1) +
               nbf_ * nbf_;
    }
    auto ref_normal() const { return ref_normal_; }

    void begin_preparation(std::size_t numFaultFaces);
    void prepare(std::size_t faultNo, FacetInfo const& info, LinearAllocator<double>& scratch);
    void end_preparation() {}
    //auto getNQ() {return quad_rule_.size();}

protected:
    std::shared_ptr<Curvilinear<DomainDimension>> cl_;
    SimplexQuadratureRule<DomainDimension - 1u> quad_rule_;
    std::array<double, DomainDimension> up_;
    std::array<double, DomainDimension> ref_normal_;

    // Basis
    std::size_t nbf_;
    std::vector<Managed<Tensor<double, 3u>>> geoDxi_q;
    Managed<Matrix<double>> e_q;
    Managed<Matrix<double>> e_q_T;

    struct SignFlipped {
        using type = bool;
    };
    struct Normal {
        using type = std::array<double, DomainDimension>;
    };
    struct NormalLength {
        using type = double;
    };
    struct FaultBasis {
        using type = std::array<double, DomainDimension * DomainDimension>;
    };
    struct MInv {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    using fault_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, SignFlipped, Normal, NormalLength, FaultBasis>;
    mneme::StridedView<fault_t> fault_;

    using mass_t = mneme::MultiStorage<mneme::DataLayout::SoA, MInv>;
    mneme::StridedView<mass_t> mass_;
};

} // namespace tndm

#endif // ADAPTERBASE_20210906_H
