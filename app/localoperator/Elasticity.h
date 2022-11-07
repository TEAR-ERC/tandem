#ifndef ELASTICITY_20200929_H
#define ELASTICITY_20200929_H

#include "basis/WarpAndBlend.h"
#include "config.h"
#include "localoperator/NodalInterpolation.h"

#include "form/DGCurvilinearCommon.h"
#include "form/FacetInfo.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include "mneme/allocators.hpp"
#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tndm {

class Elasticity : public DGCurvilinearCommon<DomainDimension> {
public:
    using base = DGCurvilinearCommon<DomainDimension>;
    constexpr static std::size_t Dim = DomainDimension;
    constexpr static std::size_t NumQuantities = DomainDimension;

    Elasticity(std::shared_ptr<Curvilinear<DomainDimension>> cl, std::vector<int> const& regions, region_functional_t<1> lam,
               region_functional_t<1> mu, std::optional<region_functional_t<1>> rho = std::nullopt,
               DGMethod method = DGMethod::IP);

    constexpr std::size_t alignment() const { return ALIGNMENT; }
    std::size_t block_size() const { return space_.numBasisFunctions() * NumQuantities; }
    auto make_interpolation_op() const {
        return std::make_unique<NodalInterpolation<Dim>>(
            PolynomialDegree, WarpAndBlendFactory<Dim>(), NumQuantities, alignment());
    }

    void begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                           std::size_t numLocalFacets);
    void prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_penalty(std::size_t fctNo, FacetInfo const& info, LinearAllocator<double>&);
    void prepare_cfl(std::size_t elNo, mneme::span<SideInfo> info, LinearAllocator<double>&);

    bool assemble_volume(std::size_t elNo, Matrix<double>& A00,
                         LinearAllocator<double>& scratch) const;
    bool assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                           LinearAllocator<double>& scratch) const;
    bool assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           LinearAllocator<double>& scratch) const;

    bool rhs_volume(std::size_t elNo, Vector<double>& B, LinearAllocator<double>& scratch) const;
    bool rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      Vector<double>& B1, LinearAllocator<double>& scratch) const;
    bool rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      LinearAllocator<double>& scratch) const;

    void apply(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
               std::array<Vector<double const>, NumFacets> const& x_n, Vector<double>& y_0) const;
    void wave_rhs(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
                  std::array<Vector<double const>, NumFacets> const& x_n,
                  Vector<double>& y_0) const;
    void project(std::size_t elNo, volume_functional_t x, Vector<double>& y) const;

    std::size_t flops_apply(std::size_t elNo, mneme::span<SideInfo> info) const;

    TensorBase<Matrix<double>> tractionResultInfo() const;
    void traction_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Vector<double const>& u1, Matrix<double>& result) const;
    void traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Matrix<double>& result) const;

    inline double cfl_time_step(std::size_t elNo) const { return cfl_dt_[elNo]; }

    FiniteElementFunction<DomainDimension> solution_prototype(std::size_t numLocalElements) const {
        auto names = std::vector<std::string>(NumQuantities);
        char buf[100];
        for (std::size_t q = 0; q < NumQuantities; ++q) {
            snprintf(buf, sizeof(buf), "u%lu", q);
            names[q] = buf;
        }
        return FiniteElementFunction<DomainDimension>(space_.clone(), names, numLocalElements);
    }

    FiniteElementFunction<DomainDimension>
    coefficients_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension>(materialSpace_.clone(), {"lambda", "mu"},
                                                      numLocalElements);
    }
    void coefficients_volume(std::size_t elNo, Matrix<double>& C, LinearAllocator<double>&) const;

    void set_force(functional_t<NumQuantities> fun) {
        fun_force = make_volume_functional(std::move(fun));
    }
    void set_force(volume_functional_t fun) { fun_force = std::move(fun); }
    void set_dirichlet(functional_t<NumQuantities> fun) {
        fun_dirichlet = make_facet_functional(std::move(fun));
    }
    void set_dirichlet(functional_t<NumQuantities> fun,
                       std::array<double, DomainDimension> const& refNormal) {
        fun_dirichlet = make_facet_functional(std::move(fun), refNormal);
    }
    void set_dirichlet(facet_functional_t fun) { fun_dirichlet = std::move(fun); }
    void set_slip(functional_t<NumQuantities> fun,
                  std::array<double, DomainDimension> const& refNormal) {
        fun_slip = make_facet_functional(std::move(fun), refNormal);
    }
    void set_slip(facet_functional_t fun) { fun_slip = std::move(fun); }

private:
    template <bool WithRHS>
    void apply_(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
                std::array<Vector<double const>, NumFacets> const& x_n, Vector<double>& y_0) const;

    double penalty(std::size_t fctNo) const {
        if (method_ == DGMethod::BR2) {
            return NumFacets;
        }
        return penalty_[fctNo];
    }
    std::pair<double, double> stiffness_tensor_bounds(std::size_t elNo) const;
    double inverse_density_upper_bound(std::size_t elNo) const;
    void compute_mass_matrix(std::size_t elNo, double* M) const;
    void compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const;
    bool bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const;
    bool bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const;
    void transpose_JInv(std::size_t fctNo, int side);

    DGMethod method_;

    // Ref elements
    NodalRefElement<DomainDimension> space_;
    NodalRefElement<DomainDimension> materialSpace_;

    // Matrices
    Managed<Matrix<double>> MhatInv;
    Managed<Matrix<double>> E_Q;
    Managed<Matrix<double>> E_Q_T;
    Managed<Matrix<double>> negative_E_Q_T;
    Managed<Tensor<double, 3u>> Dxi_Q;
    Managed<Tensor<double, 3u>> Dxi_Q_120;
    std::vector<Managed<Matrix<double>>> E_q;
    std::vector<Managed<Matrix<double>>> E_q_T;
    std::vector<Managed<Matrix<double>>> negative_E_q;
    std::vector<Managed<Matrix<double>>> negative_E_q_T;
    std::vector<Managed<Tensor<double, 3u>>> Dxi_q;
    std::vector<Managed<Tensor<double, 3u>>> Dxi_q_120;

    Managed<Matrix<double>> matE_Q_T;
    std::vector<Managed<Matrix<double>>> matE_q_T;

    // Input
    volume_functional_t fun_lam;
    volume_functional_t fun_mu;
    volume_functional_t fun_rho;
    std::optional<volume_functional_t> fun_force = std::nullopt;
    std::optional<facet_functional_t> fun_dirichlet = std::nullopt;
    std::optional<facet_functional_t> fun_slip = std::nullopt;

    // Precomputed data
    struct lam {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct rhoInv {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct lam_W_J_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu_W_J_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct negative_rhoInv_W_Jinv_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct lam_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct lam_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct JInvT {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct JInvT0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct JInvT1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    using material_vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, lam, mu, rhoInv>;
    mneme::StridedView<material_vol_t> material;

    using vol_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, lam_W_J_Q, mu_W_J_Q,
                                          negative_rhoInv_W_Jinv_Q, JInvT>;
    mneme::StridedView<vol_pre_t> volPre;

    using fct_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, lam_q_0, mu_q_0, lam_q_1, mu_q_1,
                                          JInvT0, JInvT1>;
    mneme::StridedView<fct_pre_t> fctPre;

    std::vector<double> penalty_;
    std::vector<double> cfl_dt_;

    // Options
    constexpr static double epsilon = -1.0;
};

} // namespace tndm

#endif // ELASTICITY_20200929_H
