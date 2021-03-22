#ifndef SEASELASTICITYADAPTER_20201103_H
#define SEASELASTICITYADAPTER_20201103_H

#include "common/MGConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscVector.h"
#include "geometry/Curvilinear.h"
#include "localoperator/Elasticity.h"
#include "tandem/SeasAdapterBase.h"

#include "form/BoundaryMap.h"
#include "form/DGOperator.h"
#include "form/DGOperatorTopo.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"
#include "util/LinearAllocator.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace tndm {

class SeasElasticityAdapter : public SeasAdapterBase {
public:
    using local_operator_t = Elasticity;
    constexpr static std::size_t Dim = local_operator_t::Dim;
    constexpr static std::size_t NumQuantities = local_operator_t::NumQuantities;
    using time_functional_t =
        std::function<std::array<double, NumQuantities>(std::array<double, Dim + 1u> const&)>;

    SeasElasticityAdapter(std::shared_ptr<Curvilinear<Dim>> cl,
                          std::shared_ptr<DGOperatorTopo> topo,
                          std::unique_ptr<RefElement<Dim - 1u>> space,
                          std::unique_ptr<local_operator_t> local_operator,
                          std::array<double, Dim> const& up,
                          std::array<double, Dim> const& ref_normal,
                          MGConfig const& mg_config = MGConfig());

    void set_boundary(time_functional_t fun) { fun_boundary = std::move(fun); }

    template <typename BlockVector> void solve(double time, BlockVector& state) {
        auto in_handle = state.begin_access_readonly();
        dgop_->lop().set_slip(
            [this, &state, &in_handle](std::size_t fctNo, Matrix<double>& f_q, bool) {
                auto faultNo = this->faultMap_.bndNo(fctNo);
                auto state_block = state.get_block(in_handle, faultNo);
                this->slip(faultNo, state_block, f_q);
            });
        dgop_->lop().set_dirichlet(
            [this, time](std::array<double, Dim> const& x) {
                std::array<double, Dim + 1u> xt;
                std::copy(x.begin(), x.end(), xt.begin());
                xt.back() = time;
                return this->fun_boundary(xt);
            },
            ref_normal_);
        linear_solver_.update_rhs(*dgop_);
        linear_solver_.solve();
        state.end_access_readonly(in_handle);
    }

    TensorBase<Matrix<double>> traction_info() const;
    template <class Func> void begin_traction(Func state_access) {
        handle_ = linear_solver_.x().begin_access_readonly();
        dgop_->lop().set_slip([this, state_access](std::size_t fctNo, Matrix<double>& f_q, bool) {
            auto faultNo = this->faultMap_.bndNo(fctNo);
            auto state_block = state_access(faultNo);
            this->slip(faultNo, state_block, f_q);
        });
    }
    void traction(std::size_t faultNo, Matrix<double>& traction, LinearAllocator<double>&) const;
    void end_traction() { linear_solver_.x().end_access_readonly(handle_); }

    auto displacement() const { return dgop_->solution(linear_solver_.x()); }
    std::size_t numLocalElements() const { return dgop_->numLocalElements(); }
    std::size_t number_of_local_dofs() const { return dgop_->number_of_local_dofs(); }

private:
    void slip(std::size_t faultNo, Vector<double const>& state, Matrix<double>& s_q) const;

    std::unique_ptr<DGOperator<local_operator_t>> dgop_;
    PetscLinearSolver linear_solver_;

    time_functional_t fun_boundary =
        [](std::array<double, Dim + 1u> const& x) -> std::array<double, NumQuantities> {
        return {};
    };
    PetscVector::const_handle handle_;
};

} // namespace tndm

#endif // SEASELASTICITYADAPTER_20201103_H
