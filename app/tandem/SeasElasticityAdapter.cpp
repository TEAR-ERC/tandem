#include "SeasElasticityAdapter.h"

#include "kernels/elasticity/tensor.h"
#include "kernels/elasticity_adapter/kernel.h"
#include "kernels/elasticity_adapter/tensor.h"

#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "localoperator/Elasticity.h"
#include "tensor/Managed.h"

#include <cassert>

namespace tndm {
SeasElasticityAdapter::SeasElasticityAdapter(std::shared_ptr<DGOperatorTopo> topo,
                                             std::unique_ptr<RefElement<Dim - 1u>> space,
                                             std::unique_ptr<Elasticity> local_operator,
                                             std::array<double, Dim> const& ref_normal,
                                             double normal_stress)
    : SeasAdapterBase(topo, std::move(space), local_operator->facetQuadratureRule().points(),
                      ref_normal),
      dgop_(std::make_unique<DGOperator<Elasticity>>(std::move(topo), std::move(local_operator))),
      linear_solver_(*dgop_), normal_stress_(normal_stress) {}

void SeasElasticityAdapter::slip(std::size_t faultNo, Vector<double const>& state,
                                 Matrix<double>& slip_q) const {
    auto const nbf = space_->numBasisFunctions();
    double slip_flip[elasticity_adapter::tensor::slip::Size];
    assert(elasticity_adapter::tensor::slip::Size == nbf);

    for (std::size_t i = 0; i < nbf; ++i) {
        if (!sign_[faultNo].template get<SignFlipped>()[i]) {
            slip_flip[i] = -state(i);
        } else {
            slip_flip[i] = state(i);
        }
    }
    assert(slip_q.shape(0) == 1);
    assert(slip_q.shape(1) == elasticity_adapter::tensor::slip_q::size());
    elasticity_adapter::kernel::evaluate_slip krnl;
    krnl.e_q = e_q.data();
    krnl.slip = slip_flip;
    krnl.slip_q = slip_q.data();
    krnl.execute();
}

TensorBase<Matrix<double>> SeasElasticityAdapter::traction_info() const {
    return TensorBase<Matrix<double>>(elasticity_adapter::tensor::traction::Shape[0], 2);
}

void SeasElasticityAdapter::traction(std::size_t faultNo, Matrix<double>& traction,
                                     LinearAllocator&) const {
    auto const nbf = space_->numBasisFunctions();

    double traction_q_raw[elasticity::tensor::traction_q::Size];
    auto traction_q = Matrix<double>(traction_q_raw, dgop_->lop().tractionResultInfo());
    assert(traction_q.size() == elasticity::tensor::traction_q::Size);

    auto fctNo = faultMap_.fctNo(faultNo);
    auto const& info = dgop_->topo().info(fctNo);
    auto u0 = linear_solver_.x().get_block(handle_, info.up[0]);
    auto u1 = linear_solver_.x().get_block(handle_, info.up[1]);
    dgop_->lop().traction(fctNo, info, u0, u1, traction_q);
    elasticity_adapter::kernel::evaluate_traction krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.traction_q = traction_q_raw;
    krnl.minv = minv.data();
    krnl.traction = &traction(0, 1);
    //krnl.unit_normal = sign_[faultNo].template get<UnitNormal>().data()->data();
    krnl.w = dgop_->lop().facetQuadratureRule().weights().data();
    krnl.execute();

    for (std::size_t i = 0; i < nbf; ++i) {
        traction(i, 0) = normal_stress_;
    }
}

} // namespace tndm
